# Databricks notebook source
# MAGIC %md
# MAGIC # MyFixit Dataset
# MAGIC
# MAGIC This repository contains the MyFixit dataset. It also includes the processed data and column corpus required for the [MyFixit Annotator](https://github.com/rub-ksv/MyFixit-Annotator). 
# MAGIC
# MAGIC MyFixit is a collection of repair manuals, collected from [iFixit](https://www.ifixit.com) website. There are in total **31,601** repair manuals in 15 device categories. Each step in the manuals of the 'Mac Laptop' category is annotated with the required tool, disassembled parts, and the removal verbs (in total **1,497** manuals with **36,659** steps). The rest of the categories do not have human annotations yet.
# MAGIC
# MAGIC For the details of dataset and the annotation guideline, please refer to the [paper](http://www.lrec-conf.org/proceedings/lrec2020/pdf/2020.lrec-1.260.pdf) published in LREC 2020.
# MAGIC
# MAGIC Here is an example of an annotated step in the dataset:
# MAGIC
# MAGIC     {"Title": "MacBook Unibody Model A1278 Hard Drive Replacement", 
# MAGIC     "Ancestors": ["MacBook", "Mac Laptop", "Mac", "Root"], 
# MAGIC     "Guidid": 816, 
# MAGIC     "Category": "MacBook Unibody Model A1278", 
# MAGIC     "Subject": "Hard Drive",
# MAGIC     "Toolbox": 
# MAGIC         [{"Name": ["phillips 00 screwdriver"], "Url": "https://www.ifixit.com/Store/Parts/Phillips-00-Screwdriver/IF145-006", "Thumbnail": "https://da2lh5cs8ikqj.cloudfront.net/cart-products/rLfPqcRxAVqNxfwc.mini"},
# MAGIC         {"Name": ["spudger"], "Url": "http://www.ifixit.com/Tools/Spudger/IF145-002", "Thumbnail": "https://da2lh5cs8ikqj.cloudfront.net/cart-products/fIQ3oZSjd1yLgqpX.mini"},
# MAGIC         {"Name": ["t6 torx screwdriver"], "Url": "https://www.ifixit.com/Store/Tools/TR6-Torx-Security-Screwdriver/IF145-225", "Thumbnail": ""}],
# MAGIC     "Url": "https://www.ifixit.com/Guide/MacBook+Unibody+Model+A1278+Hard+Drive+Replacement/816",
# MAGIC     "Steps": [{
# MAGIC         "Order": 1,
# MAGIC         "Tools_annotated": ["NA"],
# MAGIC 		"Tools_extracted": ["NA"],
# MAGIC         "Word_level_parts_raw": [{"name": "battery", "span": [19, 19]}],
# MAGIC 		"Word_level_parts_clean": ["battery"],
# MAGIC         "Removal_verbs": [{"name": "pull out", "span": [17, 17], "part_index": [0]}], 
# MAGIC         "Lines":
# MAGIC             [{"Text": "be sure the access door release latch is vertical before proceeding."},
# MAGIC             {"Text": "grab the white plastic tab and pull the battery up and out of the unibody."}],
# MAGIC         "Text_raw": "Be sure the access door release latch is vertical before proceeding. Grab the white plastic tab and pull the battery up and out of the Unibody.",
# MAGIC         "Images": ["https://d3nevzfk7ii3be.cloudfront.net/igi/WkwQip2DfR1iJLMX.standard"], 
# MAGIC         "StepId": 4122},
# MAGIC          ...]}
# MAGIC
# MAGIC **The data and models are available in the following formats:**
# MAGIC
# MAGIC ## 1- Complete dataset
# MAGIC There are 15 category of manuals. Here are the statistics of each category:
# MAGIC
# MAGIC | Category          | Number of manuals | Number of steps with unique text |
# MAGIC |-------------------|-------------------|----------------------------------|
# MAGIC | Mac               | 2868              | 8893                             |
# MAGIC | Car and Truck     | 761               | 3320                             |
# MAGIC | Household         | 1710              | 7859                             |
# MAGIC | Computer Hardware | 927               | 4502                             |
# MAGIC | Appliance         | 1333              | 5744                             |
# MAGIC | Camera            | 2761              | 12000                            |
# MAGIC | PC                | 6677              | 26181                            |
# MAGIC | Electronics       | 2343              | 9765                             |
# MAGIC | Phone             | 6005              | 20573                            |
# MAGIC | Game Console      | 1008              | 4517                             |
# MAGIC | Skills            | 140               | 885                              |
# MAGIC | Vehicle           | 374               | 1815                             |
# MAGIC | Media Player      | 649               | 2697                             |
# MAGIC | Apparel           | 382               | 2051                             |
# MAGIC | Tablet            | 2756              | 10679                            |
# MAGIC
# MAGIC For each category, there is a JSON file that contains all the collected manuals with more than one step and one tool. The teardown manuals are excluded from the data. 
# MAGIC
# MAGIC The JSON files are collections of JSON-like objects with one object in each line.
# MAGIC
# MAGIC ### Finding the relevant manuals:
# MAGIC There is a simple script [search.py](search.py) that helps you to find the proper manuals and save them in XML or JSON format. The script receives the following arguments:
# MAGIC
# MAGIC     -device: Name of the device. (Optional)
# MAGIC     -input: Name of one of the files in 'jsons/' directory (Required)
# MAGIC     -part: Part of the device to repair. (Optional)
# MAGIC     -format: The format of output data, XML or JSON. (Optional, default is JSON)
# MAGIC     -output: Name of the output file. (Required)
# MAGIC     -mintools: Minimum number of tools in the manual. (Optional)
# MAGIC     -minsteps: Minimum number of steps in the manual. (Optional)
# MAGIC     -verbose: Prints the titles of selected manuals. (Optional)
# MAGIC     -annotatedtool: Only selecting the manuals with the annotation of required tools (Optional)
# MAGIC     -annotatedpart: Only selecting the manuals with the annotation of disassembled parts (Optional)
# MAGIC
# MAGIC Required libraries:
# MAGIC > dicttoxml (only if xml is selected as the output format)    
# MAGIC
# MAGIC Example:
# MAGIC
# MAGIC     python search.py -input Mac.json -output tmp -device macbook pro -part battery -mintools 2 -minsteps 15 -format xml -verbose -annotatedtool -annotatedpart
# MAGIC     
# MAGIC Output:
# MAGIC
# MAGIC     Total number of matched manuals :29  
# MAGIC     Title of manuals:  
# MAGIC     MacBook Pro 17" Models A1151 A1212 A1229 and A1261 Battery Connector Replacement  
# MAGIC     MacBook Pro 17" Models A1151 A1212 A1229 and A1261 PRAM Battery Replacement  
# MAGIC     MacBook Pro 15" Core 2 Duo Model A1211 PRAM Battery Replacement  
# MAGIC     MacBook Pro 15" Core 2 Duo Model A1211 Battery Connector Replacement  
# MAGIC     MacBook Pro 15" Core Duo Model A1150 PRAM Battery Replacement  
# MAGIC     MacBook Pro 15" Core Duo Model A1150 Battery Connector Replacement  
# MAGIC     MacBook Pro 15" Core 2 Duo Models A1226 and A1260 Battery Connector Replacement  
# MAGIC     MacBook Pro 15" Unibody Late 2008 and Early 2009 Battery Connector Replacement  
# MAGIC     MacBook Pro 13" Retina Display Late 2012 Battery Replacement  
# MAGIC     MacBook Pro 13" Retina Display Early 2013 Battery Replacement  
# MAGIC     MacBook Pro 13" Retina Display Late 2013 Battery Replacement  
# MAGIC     MacBook Pro 13" Retina Display Mid 2014 Battery Replacement  
# MAGIC     MacBook Pro 13" Retina Display Early 2015 Battery Replacement  
# MAGIC     MacBook Pro 13" Function Keys Late 2016 Battery Replacement  
# MAGIC     MacBook Pro 15" Retina Display Mid 2012 Battery Replacement  
# MAGIC     MacBook Pro 15" Retina Display Late 2013 Battery Replacement  
# MAGIC     MacBook Pro 15" Retina Display Mid 2015 Battery Replacement  
# MAGIC     MacBook Pro 15" Retina Display Early 2013 Battery Replacement  
# MAGIC     MacBook Pro 15" Retina Display Mid 2014 Battery Replacement  
# MAGIC     MacBook Pro 13" Retina Display Late 2012 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 13" Retina Display Early 2013 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 13" Retina Display Late 2013 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 13" Retina Display Mid 2014 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 13" Retina Display Early 2015 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 15" Retina Display Mid 2012 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 15" Retina Display Late 2013 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 15" Retina Display Mid 2014 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 15" Retina Display Early 2013 Battery Replacement (Legacy)  
# MAGIC     MacBook Pro 15" Retina Display Mid 2015 Battery Replacement (Legacy)  
# MAGIC     Selected manuals are saved in tmp.xml
# MAGIC
# MAGIC ### Running a Mongodb sever and importing data:
# MAGIC
# MAGIC To work with the annotator tool, you need to load the database into a running MongoDB server.  
# MAGIC
# MAGIC For learning about mongodb installation please refer to its [documentations](https://docs.mongodb.com/manual/installation/).
# MAGIC
# MAGIC #### After running the sever you can import the dataset  with following command:
# MAGIC
# MAGIC     mongoimport --db myfixit --collection posts --file <fileName>.json
# MAGIC
# MAGIC   
# MAGIC
# MAGIC ## 2- Processed data for the [MyFixit annotator](https://github.com/rub-ksv/MyFixit-Annotator)
# MAGIC
# MAGIC The web-based MyFixit annotator produces a table of processed data for the selected device category and fills the annotation pages with it. The tables include the extracted tools, cleaned text, the annotated and unannotated sentences separated in each step, the part and verb candidates extracted either with the unsupervised (basic) model or the supervised (deep) model. The tables also have the list of parts' names that were filtered by Wordnet at each step.
# MAGIC
# MAGIC In order to work with the annotator without running the parser/tagger, copy the tables to `/src/web_app/static/tables/`. Otherwise, the tables will be generated automatically from the database. Generating the tables might take some time, depending on the size of the selected category and the chosen model. 
# MAGIC
# MAGIC ## 3- Column corpus of annotated data
# MAGIC
# MAGIC If the table of processed data does not exist for the selected category, and if the supervised method is selected for part extraction, the app trains a bilstm-CRF based model from the annotated data. For doing that, it looks for a column corpus in `/src/part_extraction/data/`, in which the labels are represented in the BIEO form. If the file does not exist, it will be automatically produced from the annotated steps in the database. There is a column corpus for the category Mac laptop.
# MAGIC
# MAGIC # Cite
# MAGIC If you found this dataset or our work useful, please cite:
# MAGIC
# MAGIC     @InProceedings{nabizadeh-kolossa-heckmann:2020:LREC,
# MAGIC       author    = {Nabizadeh, Nima  and  Kolossa, Dorothea  and  Heckmann, Martin},
# MAGIC       title     = {MyFixit: An Annotated Dataset, Annotation Tool, and Baseline Methods for Information Extraction from Repair Manuals},
# MAGIC       booktitle      = {Proceedings of The 12th Language Resources and Evaluation Conference},
# MAGIC       month          = {May},
# MAGIC       year           = {2020},
# MAGIC       address        = {Marseille, France},
# MAGIC       publisher      = {European Language Resources Association},
# MAGIC       pages     = {2120--2128}}
# MAGIC

# COMMAND ----------

# DBTITLE 1,Depdencies
import dlt
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.functions import from_json, col, when, pandas_udf, concat, lit
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, ArrayType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read from source

# COMMAND ----------

#Read from the source JSON files. These were downloaded from Kaggle and previously stored in a Databricks Volume
@dlt.table()
def manuals_raw():
  return (spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .load("/Volumes/ademianczuk/myfixit/articles/jsons")
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parse the Steps and Tools columns for easy data access

# COMMAND ----------

#Format the columns that outline the tools and steps. This will allow us to iterate and select values independently.
@dlt.table()
def manuals_bronze():

    #Define the toolbox schema
    toolbox_schema = ArrayType(
        StructType([
            StructField("Name", StringType(), True),
            StructField("Url", StringType(), True),
            StructField("Thumbnail", StringType(), True)
        ])
    )

    #Define the steps schema (This contains a nested JSON object that we'll also need to parse)
    steps_schema = ArrayType(
        StructType([
            StructField("Order", IntegerType(), True),
            StructField("Lines", ArrayType(
                StructType([
                    StructField("Text", StringType(), True)
                ])), True),
            StructField("Text_raw", StringType(), True),
            StructField("Images", ArrayType(StringType()), True),
            StructField("StepId", IntegerType(), True),
            StructField("Tools_extracted", ArrayType(StringType()), True)
        ])
    )

    #Extract and parse the columns based on the schemas and ancestor parent category
    df = spark.readStream.table("LIVE.manuals_raw")
    df_parsed = (df.withColumn("Toolbox", from_json(col("Toolbox"), toolbox_schema))
                 .withColumn("Steps", from_json(col("Steps"), steps_schema))
                 .withColumn("AncestorsArray", F.from_json(F.col("Ancestors"), ArrayType(StringType())))
                 .withColumn("Parent_Category", F.expr("AncestorsArray[size(AncestorsArray) - 2]"))
                )

    return df_parsed

# COMMAND ----------

# MAGIC %md
# MAGIC ##Coalesce the steps into a cleaned column without the extra formatting.

# COMMAND ----------

@dlt.table
def manuals_silver():

    #Create a user-defined function to coalesce the steps into a single body of text based on the 'Order' key.
    #The Pandas UDF needs to be within context of the dlt table. Otherwise the table will re-materialize recursively, thus wasting time and resources.
    
    @pandas_udf(StringType())
    def coalesce_steps_pandas_udf(steps_series: pd.Series) -> pd.Series:
        def process_steps(steps):
            
            # Parse and process the steps. Sort by the 'Order' key before joining the elements.
            sorted_steps = sorted(steps, key=lambda x: int(x['Order']))
            return " ".join(step["Text_raw"] for step in sorted_steps)
        
        #Apply processing to each row
        return steps_series.apply(process_steps)
    
    #Source the upstream table as a materialization
    df = (spark.readStream.table("LIVE.manuals_bronze")
          .withColumn("Subject", when(col("Subject")=="" , "Generic").otherwise(col("Subject")))
          )
    df = df.withColumn("coalesced_steps", coalesce_steps_pandas_udf(df["Steps"]))
    
    return df

# COMMAND ----------

#Add the title of the article for context-relevant metadata. We will be enriching this later when we start building our our agents and experts.

@dlt.table
def manuals_silver_rag_prep():

    df = spark.readStream.table("LIVE.manuals_silver")
    df = (df.withColumn("article", concat(lit("Title: "), df["Title"], lit('\r\n'), lit("Steps: "), df["coalesced_steps"])))

    return df
