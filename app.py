
# Run it with /Users/espina/Unsynced/spark-3.1.2-bin-hadoop3.2/bin/spark-submit "/Users/espina/Documents/MSc Data Science/Big Data: Data Visualization/spark_assignment/app.py"

import sys
from pyspark_dist_explore import hist
import pandas as pd
import matplotlib.pyplot as plt
 
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
 
if __name__ == "__main__":

    path = "/Users/espina/Unsynced/Datasets/dataverse_files/2007.csv.bz2"

    # Create Spark context with Spark configuration
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

    # Read the dataset into a spark dataframe
    df = spark.read.csv(path, header=True)
    df.printSchema()
    #print(data.count())
    # print(df.schema.names)

    # Drop the forbidden columns
    # 1. Several variables may not contain useful information or are forbidden. These need to be filtered out.
    forbidden_vars = ('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay',
     'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')
    df = df.drop(*forbidden_vars)
    df.printSchema()

    # 2. Several variables may contain information in a way that is difficult to understand/process. 
    # These need to be transformed into something meaningful.

    # 3. Several variables may provide better information when combined with others. In these cases, 
    # new variables could be derived from them.