# Databricks notebook source
# MAGIC %md
# MAGIC # RAG VectorDB Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load the table with the feature column containing the article chunks
# MAGIC
# MAGIC ### A quick note on chunks
# MAGIC Our articles are actually quite small - for this project we don't need to chunk the corpus of text any more. We're not losing anything by over-generalizing since the steps are generally summarized in chunks of fewer than 1000 tokens. There are trade-offs to having both large and small chunks; small chunks can result in very precise and specific answers, but lose context. Conversely large chunks can have better context but lose specificity. Chunks that break up large bodies of text also may require some degree of overlap to 'bridge' context.

# COMMAND ----------

# MAGIC %pip install -U --quiet transformers==4.30.2 "unstructured[pdf,docx]==0.10.30" langchain==0.1.16 llama-index==0.9.3 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.9.0 flashrank==0.2.0
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Vector Search Endpoint and Index
# MAGIC
# MAGIC ### Setup a Vector Search Endpoint
# MAGIC
# MAGIC The first step for creating a Vector Search index is to create a compute endpoint. This endpoint serves the vector search index. You can query and update the endpoint using the REST API or the SDK. 

# COMMAND ----------

# assign vs search endpoint by username
vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name =  "vs_general"

print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

vsc = VectorSearchClient(disable_notice=True)

# COMMAND ----------

#At least one endpoint needs to exist for this check to work
if vs_endpoint_name not in [e['name'] for e in vsc.list_endpoints()['endpoints']]:
    vsc.create_endpoint(name=vs_endpoint_name, endpoint_type="STANDARD")

# COMMAND ----------

df = spark.table("ademianczuk.myfixit.manuals_silver_rag_prep")
df = df.select("Guidid", "article")

df.write.mode("overwrite").saveAsTable("ademianczuk.myfixit.articles")

# COMMAND ----------

spark.sql("ALTER TABLE ademianczuk.myfixit.articles SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

source_table = "ademianczuk.myfixit.articles"
pk = "Guidid"
source_col = "article"
index_name = "ademianczuk.myfixit.manuals_index"

index = vsc.create_delta_sync_index(
    endpoint_name=vs_endpoint_name,
    source_table_name=source_table,
    index_name=index_name,
    pipeline_type="TRIGGERED",
    primary_key=pk,
    embedding_source_column=source_col,
    embedding_model_endpoint_name="databricks-bge-large-en"
)

# COMMAND ----------

#Once complete and synced, let's have a look at our index details
index.describe()

# COMMAND ----------

#Let's test our index and similary search
results = index.similarity_search(
    query_text="My iphone battery is not charging. How do I replace it?",
    columns=["article"],
    num_results=5
)

rows = results['result']['data_array']
for (article, score) in rows:
    print(article + " " + str(score))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Search for Similar Content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Lake Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC **📌 Note:** `similarity_search` also supports a filter parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).
# MAGIC

# COMMAND ----------

import mlflow.deployments

deploy_client = mlflow.deployments.get_deploy_client("databricks")
question = "How do I replace a graphics card?"
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

# COMMAND ----------

# get similar 5 documents
results = vsc.get_index(endpoint_name=vs_endpoint_name, index_name=index_name).similarity_search(
  query_vector=embeddings[0],
  columns=["Guidid", "article"],
  num_results=5)

# format result to align with reranker lib format 
passages = []
for doc in results.get('result', {}).get('data_array', []):
    new_doc = {"file": doc[0], "text": doc[1]}
    passages.append(new_doc)

print(passages)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Re-ranking Search Results
# MAGIC
# MAGIC For re-ranking the results, we will use a very light library. [**`flashrank`**](https://github.com/PrithivirajDamodaran/FlashRank) is an open-source reranking library based on SoTA cross-encoders. The library supports multiple models, and in this example we will use `rank-T5` model. 
# MAGIC
# MAGIC After re-ranking you can review the results to check if the order of the results has changed. 
# MAGIC
# MAGIC **💡Note:** Re-ranking order varies based on the model used!

# COMMAND ----------

from flashrank import Ranker, RerankRequest

ranker = Ranker(model_name="rank-T5-flan", cache_dir=f"dbfs://opt/")

rerankrequest = RerankRequest(query=question, passages=passages)
results = ranker.rerank(rerankrequest)
print (*results[:5], sep="\n\n")
