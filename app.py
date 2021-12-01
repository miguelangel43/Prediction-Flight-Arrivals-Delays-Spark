
# Run it with /Users/espina/Unsynced/spark-3.1.2-bin-hadoop3.2/bin/spark-submit "/Users/espina/Documents/MSc Data Science/Big Data: Data Visualization/Spark-Assignment/app.py"

import sys
from pyspark_dist_explore import hist
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config
 
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer

from classifiers.linear_regression import LinearRegressionClassifier
from classifiers.decision_tree import DecisionTreeClassifier
import preprocessing

 
if __name__ == "__main__":

    path = config.dataset_path

    # Create Spark context with Spark configuration
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")

    # Read the dataset into a spark dataframe
    df = spark.read.csv(path, header=True)
    # Get only a sample of the rows for faster computation
    df = df.limit(1000)
    #print(data.count())
    #print(df.schema.names)

    # Drop the forbidden columns
    forbidden_vars = ('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay',
     'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')
    df = df.drop(*forbidden_vars)
    # print('Original dataframe schema')
    # df.printSchema()

     # # Cast columns datatypes to adequate one
    # # Cast to int these numerical columns
    df = df.withColumn("Year",col("Year").cast("int"))
    df = df.withColumn("Month",col("Month").cast("int"))
    df = df.withColumn("DayofMonth",col("DayofMonth").cast("int"))
    df = df.withColumn("DayOfWeek",col("DayOfWeek").cast("int"))
    df = df.withColumn("DepTime",col("DepTime").cast("int"))
    df = df.withColumn("CRSDepTime",col("CRSDepTime").cast("int"))
    df = df.withColumn("CRSArrTime",col("CRSArrTime").cast("int"))
    df = df.withColumn("FlightNum",col("FlightNum").cast("int")) # Irrelevant variable?
    df = df.withColumn("CRSElapsedTime",col("CRSElapsedTime").cast("int"))
    df = df.withColumn("ArrDelay",col("ArrDelay").cast("int"))
    df = df.withColumn("DepDelay",col("DepDelay").cast("int"))
    df = df.withColumn("Distance",col("Distance").cast("int"))
    df = df.withColumn("TaxiOut",col("TaxiOut").cast("int"))
    df = df.withColumn("Cancelled",col("Cancelled").cast("int"))

    # Drop cancelled flights
    df = df.where("Cancelled == 0")

    # Apply StringIndexer to the categorical columns
    # cat_columns = ["UniqueCarrier", "TailNum", "Origin", "Dest", "CancellationCode"]
    # indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in cat_columns]
    # pipeline = Pipeline(stages=indexers)
    # df_r = pipeline.fit(df).transform(df)

    # print('Updated dataframe schema')
    # df.printSchema()

    # Classification
    classifier = DecisionTreeClassifier()
    sel_col = ['DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'ArrDelay']
    train_df, test_df = preprocessing.train_test_split(df, sel_col=sel_col)
    classifier.fit(train_df)
    classifier.predict(test_df)