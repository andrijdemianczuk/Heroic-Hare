# Databricks notebook source
# MAGIC %md
# MAGIC # RAG with structured data - Employee Search tool
# MAGIC
# MAGIC This demo focuses on a modern approach to enriching an LLM Foundation with structured data. As LLM frameworks and design patterns are becoming more powerful for unstructured data, we're starting to expose challenges around using structured data for context. There are a number of methodologies to address this, but we will be focusing on an approach with compound AI in mind. We will be building a simple tool and agent to ask questions from. This tool can therefore be used in other toolchains later on.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Installing and initializing our environment
# MAGIC
# MAGIC Before we get started, let's create a section of this notebook to install and initialize all of the components we need. I like to put these at the start of my notebook to make it easy to reconfigure where necessary and eliminate redundant commands.

# COMMAND ----------

#We could install these on the cluster, but that tightly couples the infrastructure and code. This also works better as we move to serverless.
%pip install faker transformers sentence_transformers
%pip install --force-reinstall databricks_vectorsearch 
%pip install --upgrade langchain-databricks langchain-community langchain databricks-sql-connector

#Restart the python interpreter
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Required external libraries

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, ArrayType
from databricks import feature_engineering
from pyspark.sql import functions as F

import mlflow 
import json
import requests
import time

#############################################################################################
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from faker import Faker
import random
from pyspark.sql.window import Window

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *

# COMMAND ----------

# MAGIC %md
# MAGIC ### API access objects

# COMMAND ----------

fe = feature_engineering.FeatureEngineeringClient()
w = WorkspaceClient()
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Storage objects

# COMMAND ----------

username = "ademianczuk"
catalog_name = username

#Fetch the username to use as the schema name.
schema_name = "rag_agent_prototype"

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog_name}.{schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

# COMMAND ----------

spark.sql(f"DROP TABLE IF EXISTS {catalog_name}.{schema_name}.structured_feature_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate data for the demo
# MAGIC We will be using synthetic data for this demo. This will be used to build a feature engineering table, synced with an online table for inference.

# COMMAND ----------


#Initialize Faker and Spark
fake = Faker()

#Define the schema for the DataFrame
schema = StructType([
    StructField("name", StringType(), True),
    StructField("age", IntegerType(), True),
    StructField("city", StringType(), True),
    StructField("income", FloatType(), True),
    StructField("has_car", StringType(), True)
])

#Generate synthetic data
def generate_fake_data(num_records):
    data = []
    for _ in range(num_records):
        record = (
            fake.name(),
            random.randint(18, 80),
            fake.city(),
            round(random.uniform(20000, 120000), 2),
            random.choice(['yes', 'no'])
        )
        data.append(record)
    return data

#Create the DataFrame
num_records = 1000
fake_data = generate_fake_data(num_records)
df = spark.createDataFrame(fake_data, schema)

#We don't care about ordering unique IDs, so let's take the easy declarative approach and use the monotonically_increasing_id() function.
# df = df.withColumn("unique_id", F.monotonically_increasing_id())

#Since we're going to run this on a cluster, we can use a window function to hold each row and assign the count to the window itself. This can get computationally expensive for large datasets, but for our purposes this will do for now. This is still the best if we need a deterministic outcome.
windowSpec = Window.orderBy(F.lit(1))
df = df.withColumn("unique_id", F.row_number().over(windowSpec))

#Show the DataFrame
df.show(10)

# We'll actually be committing this table as a feature engineering delta table. We can either convert a 
# standard delta table, or define it as soon as the dataframe is written. We'll do the latter.
# df.write.format("delta").mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.structured_feature_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a feature table

# COMMAND ----------

employee_feature_table = fe.create_table(
    name = f'{catalog_name}.{schema_name}.employee_feature_table',
    primary_keys='unique_id',
    schema=df.schema,
    df=df,
    description='Employee features'
)

# COMMAND ----------

#Reload the dataframe from the feature table we created earlier. This will help with lineage.
df = fe.read_table(name=f"{catalog_name}.{schema_name}.employee_feature_table")

# COMMAND ----------

#Enable CDF on the employee feature table
spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.employee_feature_table SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an online copy of the feature table

# COMMAND ----------

#Create user preferences online table
user_spec = OnlineTableSpec(
  primary_key_columns=["unique_id"],
  source_table_full_name=f"{catalog_name}.{schema_name}.employee_feature_table",
  run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({'triggered': 'true'})
)

#Must have the workspace client api access object in scope for this to work (w)
online_user_table_pipeline = w.online_tables.create(name=f"{catalog_name}.{schema_name}.employee_feature_online", spec=user_spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Define a function to calculate embeddings

# COMMAND ----------

def calculate_embedding(text):
    embedding_endpoint_name = "databricks-bge-large-en"
    url = f"https://{mlflow.utils.databricks_utils.get_browser_hostname()}/serving-endpoints/{embedding_endpoint_name}/invocations"
    databricks_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token

    headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
        
    data = {
        "input": text
    }
    data_json = json.dumps(data, allow_nan=True)
    
    print(f"\nCalling Embedding Endpoint: {embedding_endpoint_name}\n")
    
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()['data'][0]['embedding']

# COMMAND ----------

print(calculate_embedding("Hello World!"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a feature lookup MLFlow endpoint

# COMMAND ----------

# Import necessary classes
from databricks.feature_engineering.entities.feature_lookup import FeatureLookup
from databricks.feature_engineering import FeatureEngineeringClient, FeatureFunction
from databricks.feature_engineering.entities.feature_serving_endpoint import (
    EndpointCoreConfig,
    ServedEntity
)

# Set endpoint and feature table names
user_endpoint_name = f"{username}-employee-data"

# Create a lookup to fetch features by key
features=[
  FeatureLookup(
    table_name=f"{catalog_name}.{schema_name}.employee_feature_table",
    lookup_key="unique_id"
  )
]

# Create feature spec with the lookup for features. This is important; this is what translates the structured 
# data into something we can use to create a tool and agent from.
employee_spec_name = f"{catalog_name}.{schema_name}.employee_online_spec"
try:
  fe.create_feature_spec(name=employee_spec_name, features=features)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e
  
# Create endpoint for serving user employee data
try:
  status = fe.create_feature_serving_endpoint(
    name=user_endpoint_name, 
    config=EndpointCoreConfig(
      served_entities=ServedEntity(
        feature_spec_name=employee_spec_name, 
        workload_size="Small", 
        scale_to_zero_enabled=True)
      )
    )

  # Print endpoint creation status
  print(status)
except Exception as e:
  if "already exists" in str(e):
    pass
  else:
    raise e

# COMMAND ----------

# Get endpoint status
status = fe.get_feature_serving_endpoint(name=user_endpoint_name)
print(status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create our support tool for structured data feature lookups

# COMMAND ----------

from langchain.tools import BaseTool
from typing import Union

#Create a concrete class derived from LangChain's BaseTool class. The @abstractmethod is _run, which must be present to bootstrap the tool.
class EmployeeSearchTool(BaseTool):
    name: str = "Employee Feature Server"
    description: str = "Use this tool when you need employee information such as City, Name, Income and if they own a car for their work."

    def _run(self, unique_id: str):
        import requests
        import pandas as pd
        import json
        import mlflow

        url = f"https://{mlflow.utils.databricks_utils.get_browser_hostname()}/serving-endpoints/{user_endpoint_name}/invocations"

        databricks_token = mlflow.utils.databricks_utils.get_databricks_host_creds().token
        
        headers = {'Authorization': f'Bearer {databricks_token}', 'Content-Type': 'application/json'}
        
        data = {
            "dataframe_records": [{"name": unique_id, "unique_id": 90}]
        }
        data_json = json.dumps(data, allow_nan=True)
        
        print(f"\nCalling Feature Serving Endpoint: {user_endpoint_name}\n")
        
        response = requests.request(method='POST', headers=headers, url=url, data=data_json)
        if response.status_code != 200:
          raise Exception(f'Request failed with status {response.status_code}, {response.text}')

        return response.json()['outputs'][0]['income']
    
    def _arun(self, user_id: str):
        #Running this tool asynchronously on PySpark clusters can have unintended consequences. Trust me. Don't do it.
        raise NotImplementedError("This tool does not support async")

# COMMAND ----------

from langchain_databricks import ChatDatabricks

llm = ChatDatabricks(
    endpoint="databricks-dbrx-instruct",
    extra_params={"temperature": 0.01}
)

# COMMAND ----------

print(llm.invoke('What is Databricks?'))

# COMMAND ----------

from langchain.agents import initialize_agent

#Let's use the LangChain agent & tool framework
from langchain.agents import Tool

tools = [
  EmployeeSearchTool()
]

# tools = []

import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

# Initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

# Initialize agent with tools
aibot = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=5,
    early_stopping_method='force',
    memory=conversational_memory
)

# COMMAND ----------

sys_msg = """Assistant is a large language model trained by Databricks.

Assistant is designed to answer questions about employees.

When a question is asked about an employee, use their name to match on. All questions must be only about the salary.

Overall, Assistant is a powerful system that can help users ask for information about the company employees and provide valuable insights and information on a wide range of topics.
"""

# COMMAND ----------

new_prompt = aibot.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)

aibot.agent.llm_chain.prompt = new_prompt

# COMMAND ----------

aibot_output = aibot('what is the income of Carolyn Martin?')

# COMMAND ----------

print(aibot_output['output'])
