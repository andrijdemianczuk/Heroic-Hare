# Databricks notebook source
# MAGIC %md
# MAGIC # Tune a text classification model with Hugging Face Transformers
# MAGIC This notebook trains a SMS spam classifier with "distillibert-base-uncased" as the base model on a single GPU machine
# MAGIC using the [🤗&nbsp;Transformers](https://huggingface.co/docs/transformers/index) library.
# MAGIC
# MAGIC It first downloads a small dataset, copies it to [DBFS](https://docs.databricks.com/dbfs/index.html), then converts it to a Spark DataFrame. Preprocessing up to tokenization is done on Spark. While DBFS is used as a convenience to access the datasets directly as local files on the driver, you can modify it to avoid use of DBFS. 
# MAGIC
# MAGIC Text tokenization of the SMS messages is done in `transformers` in the model's default tokenizer in order to have consistency in tokenization with the base model. The notebook uses the [Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) utility in the `transformers` library to fine-tune the model. The notebook wraps the tokenizer and trained model in a Transformers `pipeline` and logs the pipeline as an MLflow model. 
# MAGIC This make it easy to directly apply the pipeline as a UDF on Spark DataFrame string columns.
# MAGIC
# MAGIC ## Cluster setup
# MAGIC For this notebook, Databricks recommends a single GPU cluster, such as a `g4dn.xlarge` on AWS or `Standard_NC4as_T4_v3` on Azure. You can [create a single machine cluster](https://docs.databricks.com/clusters/configure.html) using the personal compute policy or by choosing "Single Node" when creating a cluster. This notebook works with Databricks Runtime ML GPU version 11.1 or greater. Databricks Runtime ML GPU versions 9.1 through 10.4 can be used by replacing the following command with `%pip install --upgrade transformers datasets evaluate`.
# MAGIC
# MAGIC The `transformers` library is installed by default on Databricks Runtime ML. This notebook also requires [🤗&nbsp;Datasets](https://huggingface.co/docs/datasets/index) and [🤗&nbsp;Evaluate](https://huggingface.co/docs/evakyate/index), which you can install using `%pip`.

# COMMAND ----------

# MAGIC %pip install datasets evaluate

# COMMAND ----------

# MAGIC %md
# MAGIC Set up any parameters for the notebook. 
# MAGIC - The base model [DistilBERT base model (uncased)](https://huggingface.co/distilbert-base-uncased) is a great foundational model that is smaller and faster than [BERT base model (uncased)](https://huggingface.co/bert-base-uncased), but still provides similar behavior. This notebook fine tunes this base model.
# MAGIC - The `tutorial_path` sets the path in DBFS that the notebook uses to write the sample dataset. It is deleted by the last command in this notebook.
# MAGIC
# MAGIC

# COMMAND ----------

base_model = "distilbert-base-uncased" 
tutorial_path = "/FileStore/sms_tutorial" 

# COMMAND ----------

# MAGIC %md
# MAGIC # Data download and loading
# MAGIC Start by downloading the dataset and load it into a Spark DataFrame. 
# MAGIC The [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection) is available from the 
# MAGIC [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).

# COMMAND ----------

# MAGIC %sh wget -P /databricks/driver https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip

# COMMAND ----------

# MAGIC %md
# MAGIC Unzip the downloaded archive.

# COMMAND ----------

# MAGIC %sh unzip /databricks/driver/smsspamcollection.zip -d /databricks/driver/sms

# COMMAND ----------

# MAGIC %md
# MAGIC Copy the file to DBFS.

# COMMAND ----------

dbutils.fs.mkdirs(f"dbfs:{tutorial_path}")
dbutils.fs.cp("file:/databricks/driver/sms/SMSSpamCollection", f"dbfs:{tutorial_path}/SMSSpamCollection.tsv")

# COMMAND ----------

# MAGIC %md
# MAGIC Load the dataset into a DataFrame. The file is tab separated and does not contain a header, so we specify the separator using `sep` and specify the column names explicitly.

# COMMAND ----------

sms = spark.read.csv(f"{tutorial_path}/SMSSpamCollection.tsv", header=False, inferSchema=True, sep="\t").toDF("label", "text")
display(sms)
sms.count()

# COMMAND ----------

# MAGIC %md
# MAGIC # Data preparation
# MAGIC
# MAGIC The datasets passed into the transformers trainer for text classification need to have integer labels [0, 1]. 

# COMMAND ----------

# MAGIC %md
# MAGIC Collect the labels and generate a mapping from labels to IDs and vice versa. `transformers` models need
# MAGIC these mappings to correctly translate the integer values into the human readable labels.

# COMMAND ----------

labels = sms.select(sms.label).groupBy(sms.label).count().collect()
id2label = {index: row.label for (index, row) in enumerate(labels)} 
label2id = {row.label: index for (index, row) in enumerate(labels)}

# COMMAND ----------

# MAGIC %md
# MAGIC Replace the string labels with the IDs in the DataFrame.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
@pandas_udf('integer')
def replace_labels_with_ids(labels: pd.Series) -> pd.Series:
  return labels.apply(lambda x: label2id[x])

sms_id_labels = sms.select(replace_labels_with_ids(sms.label).alias('label'), sms.text)
display(sms_id_labels)

# COMMAND ----------

# MAGIC %md
# MAGIC Hugging Face `datasets` supports loading from Spark DataFrames using `datasets.Dataset.from_spark`. See the Hugging Face documentation to learn more about the [from_spark()](https://huggingface.co/docs/datasets/use_with_spark) method. 
# MAGIC
# MAGIC Dataset.from_spark caches the dataset. In this example, the model is trained on the driver, and the cached data is parallelized using Spark, so `cache_dir` must be accessible to the driver and to all the workers. You can use the Databricks File System (DBFS) root([AWS](https://docs.databricks.com/dbfs/index.html#what-is-the-dbfs-root)| [Azure](https://learn.microsoft.com/azure/databricks/dbfs/#what-is-the-dbfs-root) |[GCP](https://docs.gcp.databricks.com/dbfs/index.html#what-is-the-dbfs-root)) or mount point ([AWS](https://docs.databricks.com/dbfs/mounts.html) | [Azure](https://learn.microsoft.com/azure/databricks/dbfs/mounts) | [GCP](https://docs.gcp.databricks.com/dbfs/mounts.html)). 
# MAGIC
# MAGIC By using DBFS, you can reference "local" paths when creating the `transformers` compatible datasets used for model training.

# COMMAND ----------

(train_df, test_df) = sms_id_labels.persist().randomSplit([0.8, 0.2])

import datasets
train_dataset = datasets.Dataset.from_spark(train_df, cache_dir="/dbfs/cache/train")
test_dataset = datasets.Dataset.from_spark(test_df, cache_dir="/dbfs/cache/test")

# COMMAND ----------

# MAGIC %md
# MAGIC Tokenize and shuffle the datasets for training. Since the [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) does not need the untokenized `text` columns for training,
# MAGIC the notebook removes them from the dataset. This isn't necessary, but not removing the column results in a warning during training.
# MAGIC In this step, `datasets` also caches the transformed datasets on local disk for fast subsequent loading during model training.

# COMMAND ----------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding=False, truncation=True)

train_tokenized = train_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
test_tokenized = test_dataset.map(tokenize_function, batched=True).remove_columns(["text"])
train_dataset = train_tokenized.shuffle(seed=42)
test_dataset = test_tokenized.shuffle(seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC # Model training
# MAGIC For model training, this notebook largely uses default behavior. However, you can use the full range of 
# MAGIC metrics and parameters available to the `Trainer` to adjust your model training behavior.

# COMMAND ----------

# MAGIC %md
# MAGIC Create the evaluation metric to log. Loss is also logged, but adding other metrics such as accuracy can make modeling performance easier to understand.

# COMMAND ----------

import numpy as np
import evaluate
metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# COMMAND ----------

# MAGIC %md
# MAGIC Construct default training arguments. This is where you would set many of your training parameters, such as the learning rate.
# MAGIC Refer to [transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) 
# MAGIC for the full range of arguments you can set.

# COMMAND ----------

from transformers import TrainingArguments, Trainer
training_output_dir = "/databricks/driver/sms_trainer"
training_args = TrainingArguments(output_dir=training_output_dir, evaluation_strategy="epoch")

# COMMAND ----------

# MAGIC %md
# MAGIC Create the model to train from the base model, specifying the label mappings and the number of classes.

# COMMAND ----------

from transformers import AutoModelForSequenceClassification

#Here, we're using the transformers hugging face library to pull down the base model we want to use for our tuned model (distilbert-base-uncased).
model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2, label2id=label2id, id2label=id2label)

# COMMAND ----------

# MAGIC %md
# MAGIC Using a [data collator](https://huggingface.co/docs/transformers/main_classes/data_collator) batches input
# MAGIC in training and evaluation datasets. Using the `DataCollatorWithPadding` with defaults gives good baseline
# MAGIC performance for text classification.

# COMMAND ----------

from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer)

# COMMAND ----------

# MAGIC %md
# MAGIC Construct the trainer object with the model, arguments, datasets, collator, and metrics created above.

# COMMAND ----------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)

# COMMAND ----------

# MAGIC %md
# MAGIC Construct the [MLflow](https://mlflow.org) wrapper class to store the model as a pipeline. When loading the pipeline, this model uses the GPU if CUDA is available. This model hardwires the batchsize to use with the `transformers` pipeline. You'll want to set this with the hardware you will use
# MAGIC for inference in mind.

# COMMAND ----------

import mlflow
from tqdm.auto import tqdm
import torch

pipeline_artifact_name = "pipeline"
class TextClassificationPipelineModel(mlflow.pyfunc.PythonModel):
  
  def load_context(self, context):
    device = 0 if torch.cuda.is_available() else -1
    self.pipeline = pipeline("text-classification", context.artifacts[pipeline_artifact_name], device=device)
    
  def predict(self, context, model_input): 
    texts = model_input[model_input.columns[0]].to_list()
    pipe = tqdm(self.pipeline(texts, truncation=True, batch_size=8), total=len(texts), miniters=10)
    labels = [prediction['label'] for prediction in pipe]
    return pd.Series(labels)

# COMMAND ----------

# MAGIC %md
# MAGIC Train the model, logging metrics and results to MLflow. This task is very easy for BERT-based models. Don't be
# MAGIC surprised is the evaluation accuracy is 1 or close to 1.

# COMMAND ----------

from transformers import pipeline

model_output_dir = "/databricks/driver/sms_model"
pipeline_output_dir = "/databricks/driver/sms_pipeline"
model_artifact_path = "sms_spam_model"

with mlflow.start_run() as run:
  trainer.train()
  trainer.save_model(model_output_dir)
  pipe = pipeline("text-classification", model=AutoModelForSequenceClassification.from_pretrained(model_output_dir), batch_size=8, tokenizer=tokenizer)
  pipe.save_pretrained(pipeline_output_dir)
  mlflow.transformers.log_model(transformers_model=pipe, 
                                artifact_path=model_artifact_path, 
                                input_example="Hi there!",
                               )

# COMMAND ----------

# MAGIC %md
# MAGIC # Batch inference
# MAGIC Load the model as a UDF using MLflow and use it for batch scoring.

# COMMAND ----------

logged_model = "runs:/{run_id}/{model_artifact_path}".format(run_id=run.info.run_id, model_artifact_path=model_artifact_path)

# Load model as a Spark UDF. Override result_type if the model does not return double values.
sms_spam_model_udf = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model, result_type='string')

test = test_df.select(test_df.text, test_df.label, sms_spam_model_udf(test_df.text).alias("prediction"))
display(test)

# COMMAND ----------

# MAGIC %md
# MAGIC # Cleanup
# MAGIC Remove the files placed in DBFS.

# COMMAND ----------

dbutils.fs.rm(f"dbfs:{tutorial_path}", recurse=True)
