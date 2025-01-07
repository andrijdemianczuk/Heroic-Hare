# Databricks notebook source
df = spark.table("ademianczuk.myfixit.manuals")

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, ArrayType

toolbox_schema = ArrayType(StructType([
    StructField("Name", StringType(), True),
    StructField("Url", StringType(), True),
    StructField("Thumbnail", StringType(), True)
]))

steps_schema = ArrayType(StructType([
    StructField("Order", IntegerType(), True),
    StructField("Lines", ArrayType(StructType([StructField("Text", StringType(), True)])), True),
    StructField("Text_raw", StringType(), True),
    StructField("Images", ArrayType(StringType()), True),
    StructField("StepId", IntegerType(), True),
    StructField("Tools_extracted", ArrayType(StringType()), True),
]))

df_parsed = df.withColumn("Toolbox", from_json(col("Toolbox"), toolbox_schema)).withColumn("Steps", from_json(col("Steps"), steps_schema))
display(df_parsed)

# COMMAND ----------

# MAGIC %md
# MAGIC Now that we have a well-formatted dataframe, let's convert it to something useful in terms of a manual and instructions used for an LLM. We'll have to decide how to structure this in terms of labelling. So for example, the Subject, Title, Ancestors Category and Category columns can provide useful context for our LLM. This might also imply that a variety of experts is necessary. URL might also be handy for an external reference to to the virtual assistant.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, ArrayType

df = spark.table('ademianczuk.myfixit.manuals_bronze')
# Parse the string into an array
df_with_array = df.withColumn("AncestorsArray", F.from_json(F.col("Ancestors"), ArrayType(StringType())))

# Extract the second-to-last element
df_with_category = df_with_array.withColumn(
    "Parent_Category",
    F.expr("AncestorsArray[size(AncestorsArray) - 2]")
)

# Show the result
display(df_with_category)

# COMMAND ----------

df = spark.table("ademianczuk.myfixit.manuals_bronze")

# COMMAND ----------

display(df.groupBy("Subject").count())

# COMMAND ----------

from pyspark.sql.functions import col, when

#Fix Generic Subject
df = (df.withColumn("Subject",
       when(col("Subject")=="" , "Generic")
          .otherwise(col("Subject"))))

# COMMAND ----------

display(df.groupBy("Subject").count())

# COMMAND ----------

df = spark.table("ademianczuk.myfixit.manuals_silver")

# COMMAND ----------

display(df)

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
import pandas as pd

#using a pandas udf
@pandas_udf(StringType())
def coalesce_steps_pandas_udf(steps_series: pd.Series) -> pd.Series:
    def process_steps(steps):
        # Parse and process the steps
        sorted_steps = sorted(steps, key=lambda x: int(x['Order']))
        return " ".join(step["Text_raw"] for step in sorted_steps)
    
    #Apply processing to each row
    return steps_series.apply(process_steps)

#Apply the Pandas UDF to create the new column
df = df.withColumn("coalesced_steps", coalesce_steps_pandas_udf(df["Steps"]))

#Show the result
df.select("coalesced_steps").show(truncate=False)
