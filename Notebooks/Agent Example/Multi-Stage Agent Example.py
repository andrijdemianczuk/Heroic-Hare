# Databricks notebook source
# MAGIC %pip install --upgrade --quiet langchain-core databricks-vectorsearch langchain-community youtube_search wikipedia
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

username = "ademianczuk"
catalog_name = "ademianczuk"
schema_name = "rag_agent_prototype"
working_dir = "/tmp/"


# COMMAND ----------

from langchain.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template("Tell me about a {genre} movie which {actor} is one of the actors.")
prompt_template.format(genre="horror", actor="David Howard Thornton")

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

# play with max_tokens to define the length of the response
llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

for chunk in llm_dbrx.stream("Who is David Howard Thornton?"):
    print(chunk.content, end="\n", flush=True)

# COMMAND ----------

from langchain_community.retrievers import WikipediaRetriever
retriever = WikipediaRetriever()
docs = retriever.invoke(input="David Howard Thornton")
print(docs[0])

# COMMAND ----------

print(len(docs))

# COMMAND ----------

from langchain_community.tools import YouTubeSearchTool
tool = YouTubeSearchTool()
tool.run("Terrifier Movie Trailer")

# COMMAND ----------

print(tool.description)
print(tool.args)

# COMMAND ----------

from langchain_core.output_parsers import StrOutputParser

chain = prompt_template | llm_dbrx | StrOutputParser()
print(chain.invoke({"genre":"horror", "actor":"David Howard Thornton"}))

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import YouTubeSearchTool
from databricks.vector_search.client import VectorSearchClient
from langchain.schema.runnable import RunnablePassthrough

llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 1000)
tool_yt = YouTubeSearchTool()

prompt_template_1 = PromptTemplate.from_template(
    """You are a horror movie buff. You love horror movies especially around halloween and are know to be an expert. Avoid controversial topics whenever possible, but notify the response if something is considered sensitive.

    Question: {question}

    Answer:
    """
)

chain1 = ({"question": RunnablePassthrough()} | prompt_template_1 | llm_dbrx | StrOutputParser())
print(chain1.invoke({"question":"Who starred in 'The Thing'? Why was it so popular?"}))

# COMMAND ----------

from langchain_community.vectorstores import DatabricksVectorSearch
# vsc = VectorSearchClient()
# dais_index = vsc.get_index(vs_endpoint_name, vs_index_table_fullname)
# query = "how do I use DatabricksSQL"

# dvs_delta_sync = DatabricksVectorSearch(dais_index)
# docs = dvs_delta_sync.similarity_search(query)

# videos = tool_yt.run(docs[0].page_content)
videos = tool_yt.run("Terrifier Movie Trailer")

prompt_template_2 = PromptTemplate.from_template(
    """You will get a list of videos related to the user's question. Encourage the user to watch the videos. List videos with their YouTube links.

    List of videos: {videos}
    """
)
chain2 = ({"videos": RunnablePassthrough()} | prompt_template_2 |  llm_dbrx | StrOutputParser())

# COMMAND ----------

from langchain.schema.runnable import RunnablePassthrough
from operator import itemgetter

multi_chain = ({
    "c":chain1,
    "d": chain2
}| RunnablePassthrough.assign(d=chain2))

multi_chain.invoke({"question":"Who starred in Terrifier?", "videos":videos})
