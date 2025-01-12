# Databricks notebook source
# MAGIC %md
# MAGIC # RAG Application

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup the Retriever
# MAGIC
# MAGIC We will setup the Vector Search endpoint that we created in the previous demos as retriever. The retriever will return 2 relevant documents based on the query.
# MAGIC
# MAGIC **Note:** We are not using re-ranking in this demo for the sake of the simplicity.

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain==0.1.16 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.12.1  databricks-sdk==0.28.0 "unstructured[pdf,docx]==0.10.30"
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

vs_endpoint_fallback = "vs_endpoint_fallback"
vs_endpoint_name =  "vs_general"

print(f"Vector Endpoint name: {vs_endpoint_name}. In case of any issues, replace variable `vs_endpoint_name` with `vs_endpoint_fallback` in demos and labs.")

vs_index_fullname = f"ademianczuk.myfixit.manuals_index"

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings
from databricks.sdk import WorkspaceClient

# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")
print(f"Test embeddings: {embedding_model.embed_query('What is GenerativeAI?')[:20]}...")

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="article", embedding=embedding_model
    )
    # k defines the top k documents to retrieve
    return vectorstore.as_retriever(search_kwargs={"k": 2})


# test our retriever
vectorstore = get_retriever()
similar_documents = vectorstore.invoke("How do I remove an iPod battery?")
print(f"Relevant documents: {similar_documents}")

# COMMAND ----------

# MAGIC %md
# MAGIC #Setup the Foundation Model

# COMMAND ----------

from langchain.chat_models import ChatDatabricks

# Test Databricks Foundation LLM model
chat_model = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)
print(f"Test chat model: {chat_model.invoke('What is Generative AI?')}")

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks


TEMPLATE = """You are a virtual assistant, supporting field technicians performing repairs. You are answering questions related to fixing consumer products and repair skills. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible. If the question is not about repairs, support or skills say you are not able to answer other questions.
Use the following pieces of context to answer the question at the end:

<context>
{context}
</context>

Question: {question}

Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "question"])

# The stuff documents chain ("stuff" as in "to stuff" or "to fill") is the most straightforward of the document chains. It takes a list of documents, inserts them all into a prompt and passes that prompt to an LLM.

# This chain is well-suited for applications where documents are small and only a few are passed in for most calls.

#Stuff (as in 'to stuff something in') is outlined here, along with Map-Reduce as the other option
#https://js.langchain.com/docs/tutorials/summarization/#concepts

chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=get_retriever(),
    chain_type_kwargs={"prompt": prompt}
)

# COMMAND ----------

# question = {"query": "How many grammys does Tailor Swift have?"}
# answer = chain.invoke(question)
# print(answer)

# COMMAND ----------

question = {"query": "I have a battery that's glued to the back case of my phone. How do I separate it?"}
answer = chain.invoke(question)
print(answer)

# COMMAND ----------

# MAGIC %md
# MAGIC #Testing our rag chain 

# COMMAND ----------

eval_set = """question,ground_truth,evolution_type,episode_done
"What are the limitations of symbolic planning in task and motion planning, and how can leveraging large language models help overcome these limitations?","Symbolic planning in task and motion planning can be limited by the need for explicit primitives and constraints. Leveraging large language models can help overcome these limitations by enabling the robot to use language models for planning and execution, and by providing a way to extract and leverage knowledge from large language models to solve temporally extended tasks.",simple,TRUE
"What are some techniques used to fine-tune transformer models for personalized code generation, and how effective are they in improving prediction accuracy and preventing runtime errors? ","The techniques used to fine-tune transformer models for personalized code generation include ﬁne-tuning transformer models, adopting a novel approach called Target Similarity Tuning (TST) to retrieve a small set of examples from a training bank, and utilizing these examples to train a pretrained language model. The effectiveness of these techniques is shown in the improvement in prediction accuracy and the prevention of runtime errors.",simple,TRUE
How does the PPO-ptx model mitigate performance regressions in the few-shot setting?,"The PPO-ptx model mitigates performance regressions in the few-shot setting by incorporating pre-training and fine-tuning on the downstream task. This approach allows the model to learn generalizable features and adapt to new tasks more effectively, leading to improved few-shot performance.",simple,TRUE
How can complex questions be decomposed using successive prompting?,"Successive prompting is a method for decomposing complex questions into simpler sub-questions, allowing language models to answer them more accurately. This approach was proposed by Dheeru Dua, Shivanshu Gupta, Sameer Singh, and Matt Gardner in their paper 'Successive Prompting for Decomposing Complex Questions', presented at EMNLP 2022.",simple,TRUE
"Which entity type in Named Entity Recognition is likely to be involved in information extraction, question answering, semantic parsing, and machine translation?",Organization,reasoning,TRUE
What is the purpose of ROUGE (Recall-Oriented Understudy for Gisting Evaluation) in automatic evaluation methods?,"ROUGE (Recall-Oriented Understudy for Gisting Evaluation) is used in automatic evaluation methods to evaluate the quality of machine translation. It calculates N-gram co-occurrence statistics, which are used to assess the similarity between the candidate text and the reference text. ROUGE is based on recall, whereas BLEU is based on accuracy.",simple,TRUE
"What are the challenges associated with Foundation SSL in CV, and how do they relate to the lack of theoretical foundation, semantic understanding, and explicable exploration?","The challenges associated with Foundation SSL in CV include the lack of a profound theory to support all kinds of tentative experiments, and further exploration has no handbook. The pretrained LM may not learn the meaning of the language, relying on corpus learning instead. The models cannot reach a better level of stability and match different downstream tasks, and the primary method is to increase data, improve computation power, and design training procedures to achieve better results. The lack of theoretical foundation, semantic understanding, and explicable exploration are the main challenges in Foundation SSL in CV.",simple,TRUE
How does ChatGPT handle factual input compared to GPT-3.5?,"ChatGPT handles factual input better than GPT-3.5, with a 21.9% increase in accuracy when the premise entails the hypothesis. This is possibly related to the preference for human feedback in ChatGPT's RLHF design during model training.",simple,TRUE
What are some of the challenges in understanding natural language commands for robotic navigation and mobile manipulation?,"Some challenges in understanding natural language commands for robotic navigation and mobile manipulation include integrating natural language understanding with reinforcement learning, understanding natural language directions for robotic navigation, and mapping instructions and visual observations to actions with reinforcement learning.",simple,TRUE
"How does chain of thought prompting elicit reasoning in large language models, and what are the potential applications of this technique in neural text generation and human-AI interaction?","The context discusses the use of chain of thought prompting to elicit reasoning in large language models, which can be applied in neural text generation and human-AI interaction. Specifically, researchers have used this technique to train language models to generate coherent and contextually relevant text, and to create transparent and controllable human-AI interaction systems. The potential applications of this technique include improving the performance of language models in generating contextually appropriate responses, enhancing the interpretability and controllability of AI systems, and facilitating more effective human-AI collaboration.",simple,TRUE
"Using the given context, how can the robot be instructed to move objects around on a tabletop to complete rearrangement tasks?","The robot can be instructed to move objects around on a tabletop to complete rearrangement tasks by using natural language instructions that specify the objects to be moved and their desired locations. The instructions can be parsed using functions such as parse_obj_name and parse_position to extract the necessary information, and then passed to a motion primitive that can pick up and place objects in the specified locations. The get_obj_names and get_obj_pos APIs can be used to access information about the available objects and their locations in the scene.",reasoning,TRUE
"How can searching over an organization's existing knowledge, data, or documents using LLM-powered applications reduce the time it takes to complete worker activities?","Searching over an organization's existing knowledge, data, or documents using LLM-powered applications can reduce the time it takes to complete worker activities by retrieving information quickly and efficiently. This can be done by using the LLM's capabilities to search through large amounts of data and retrieve relevant information in a short amount of time.",simple,TRUE
"""

import pandas as pd
from io import StringIO

obj = StringIO(eval_set)
eval_df = pd.read_csv(obj)

# COMMAND ----------

display(eval_df)

# COMMAND ----------

from datasets import Dataset


test_questions = eval_df["question"].values.tolist()
test_groundtruths = eval_df["ground_truth"].values.tolist()

answers = []
contexts = []

# answer each question in the dataset
for question in test_questions:
    # save the answer generated
    chain_response = chain.invoke({"query" : question})
    answers.append(chain_response["result"])
    
    # save the contexts used
    vs_response = vectorstore.invoke(question)
    contexts.append(list(map(lambda doc: doc.page_content, vs_response)))

# construct the final dataset
response_dataset = Dataset.from_dict({
    "inputs" : test_questions,
    "answer" : answers,
    "context" : contexts,
    "ground_truth" : test_groundtruths
})

# COMMAND ----------

display(response_dataset.to_pandas())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Calcuate Evaluation Metrics
# MAGIC
# MAGIC Let's use MLflow's LLM evaluation functionality to compute some of the RAG evaluation metrics.
# MAGIC
# MAGIC As we will use a second model to judge the performance of the answer, we will need to define **a model to evaluate**. In this demo, we will use `DBRX` for evaluation. 
# MAGIC
# MAGIC The metrics to compute; `answer_similarity` and `relevance`. 
# MAGIC
# MAGIC For more information about various evaluation metrics, check [MLflow LLM evaluation documentation](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html).
# MAGIC

# COMMAND ----------

import mlflow
from mlflow.deployments import set_deployments_target
from mlflow.metrics import genai

set_deployments_target("databricks")

dbrx_answer_similarity = mlflow.metrics.genai.answer_similarity(
    model="endpoints:/databricks-dbrx-instruct"
)

dbrx_relevance = mlflow.metrics.genai.relevance(
    model="endpoints:/databricks-dbrx-instruct"   
)

# results = mlflow.evaluate(
#         data=response_dataset.to_pandas(),
#         targets="ground_truth",
#         predictions="answer",
#         extra_metrics=[dbrx_answer_similarity, dbrx_relevance],
#         evaluators="default",
#     )

# COMMAND ----------

# display(results.tables['eval_results_table'])

# COMMAND ----------

# MAGIC %md
# MAGIC #Save the model to the UC registry
# MAGIC
# MAGIC Now that our model is ready and evaluated, we can register it within our Unity Catalog schema. 
# MAGIC
# MAGIC After registering the model, you can view the model and models in the **Catalog Explorer**.

# COMMAND ----------

from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_experiment("/virtual_assistant_rag_app_experiment")
mlflow.set_registry_uri("databricks-uc")
model_name = "ademianczuk.myfixit.virtual_assistant_rag_app"

with mlflow.start_run(run_name="virtual_assistant_rag_app") as run:
    signature = infer_signature(question, answer)
    model_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever, 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
    )

# COMMAND ----------

spark.sql(f"GRANT `USE SCHEMA` ON SCHEMA ademianczuk.myfixit TO `account users`;")
spark.sql(f"GRANT SELECT ON TABLE {vs_index_fullname} TO `account users`;")

spark.sql(f"GRANT `USE SCHEMA` ON SCHEMA ademianczuk.myfixit TO `cheatkodemusic@gmail.com`;")
spark.sql(f"GRANT SELECT ON TABLE {vs_index_fullname} TO `cheatkodemusic@gmail.com`;")
