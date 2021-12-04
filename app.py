
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

from classifiers.linear_regression import LinearRegressionClass
from classifiers.decision_tree import DecisionTreeClass
from classifiers.random_forest import RandomForestClass
from classifiers.tunning import Tunning
import preprocessing
import data_analysis

from pyspark.ml.stat import Correlation
 
if __name__ == "__main__":

    path = config.dataset_path

    # Create Spark context with Spark configuration
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")

    # Read the dataset into a spark dataframe
    try:
        df = spark.read.csv(path, header=True)
    except:
        print("The specified path is not pointing a readable file")
        exit(1)
        
    preprocessing.check_data(df)



    # # Get only a sample of the rows for faster computation
    # #df = df.limit(1000)
    # #print(data.count())
    # #print(df.schema.names)

    # # Drop the forbidden columns
    # forbidden_vars = ('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay',
    #  'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')
    # df = df.drop(*forbidden_vars)
    # # print('Original dataframe schema')
    # # df.printSchema()

    # # Cast columns datatypes to adequate one
    # # Cast to int these numerical columns
    # df = df.withColumn("Year",col("Year").cast("int"))
    # df = df.withColumn("Month",col("Month").cast("int"))
    # df = df.withColumn("DayofMonth",col("DayofMonth").cast("int"))
    # df = df.withColumn("DayOfWeek",col("DayOfWeek").cast("int"))
    # df = df.withColumn("DepTime",col("DepTime").cast("int"))
    # df = df.withColumn("CRSDepTime",col("CRSDepTime").cast("int"))
    # df = df.withColumn("CRSArrTime",col("CRSArrTime").cast("int"))
    # df = df.withColumn("FlightNum",col("FlightNum").cast("int")) # Irrelevant variable?
    # df = df.withColumn("CRSElapsedTime",col("CRSElapsedTime").cast("int"))
    # df = df.withColumn("ArrDelay",col("ArrDelay").cast("int"))
    # df = df.withColumn("DepDelay",col("DepDelay").cast("int"))
    # df = df.withColumn("Distance",col("Distance").cast("int"))
    # df = df.withColumn("TaxiOut",col("TaxiOut").cast("int"))
    # df = df.withColumn("Cancelled",col("Cancelled").cast("int")) # Irrelevant variable

    # # Rename explanatory variable to 'label'
    # df = df.withColumnRenamed('ArrDelay', 'label')

    # # Drop cancelled flights
    # df = df.where("Cancelled == 0")
    # df = df.drop('Cancelled')
    # df = df.drop('CancellationCode')
    # df = df.drop('UniqueCarrier')
    # # Drop null values
    # df = df.na.drop("any")

    # # Apply StringIndexer to the categorical columns
    # cat_columns = ["TailNum", "Origin", "Dest"] # "UniqueCarrier", "CancellationCode"
    # for column in cat_columns:
    #     df = preprocessing.encode_cat_vars(df, column)
    
    # # print('Updated dataframe schema')
    # df.printSchema()
    
    # all_cols = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSElapsedTime', 'TaxiOut', 'TailNum_vector', 'Origin_vector', 'Dest_vector',
    #     'DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'label']


    # # Print some statistics    
    # #data_analysis.print_correlations(df, [col for col in all_cols if col not in ['TailNum_vector', 'Origin_vector', 'Dest_vector']])
    # data_analysis.print_stats(df, [col for col in all_cols if col not in ['TailNum_vector', 'Origin_vector', 'Dest_vector']])

    # # Feature subset selection
    # # preprocessing.select_variables(df)

    # # # Train/test split
    # # train_df, test_df = preprocessing.train_test_split(df, sel_col=all_cols)

    # # Tuning
    # tuning = Tunning(train_df)
    # lr = tuning.run_lr()
    # dt = tuning.run_dt()
    # rf = tuning.run_rf()


    # # # Classification
    # # classifier = LinearRegressionClass()
    # # classifier.fit(train_df)
    # # classifier.predict(test_df)