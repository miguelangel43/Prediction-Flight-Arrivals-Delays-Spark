
# Run it with /Users/espina/Unsynced/spark-3.1.2-bin-hadoop3.2/bin/spark-submit "/Users/espina/Documents/MSc Data Science/Big Data: Data Visualization/Spark-Assignment/app.py" "/Users/espina/Unsynced/Datasets/dataverse_files/2007.csv.bz2"

import sys
from pyspark_dist_explore import hist
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
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
 
    # Create Spark context with Spark configuration
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")

    path = sys.argv[1]
    path_testing = False
    if len(sys.argv) == 3:
        path_testing = sys.argv[2]

    # Read the dataset into a spark dataframe
    df = spark.read.csv(path, header=True)
    # Get only a sample of the rows for faster computation
    df = df.limit(1000)
    #print(data.count())
    #print(df.schema.names)

    # Drop columns, cast to adequate datatypes, drop null values and encode categorical variables
    df = preprocessing.prepare_data(df)
    
    # print('Updated dataframe schema')
    df.printSchema()
    
    all_cols = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSElapsedTime', 'TaxiOut', 'Origin_vector', 'Dest_vector',
        'DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'label']


    # Print some statistics    
    #data_analysis.print_correlations(df, [col for col in all_cols if col not in ['TailNum_vector', 'Origin_vector', 'Dest_vector']])
    data_analysis.print_stats(df, [col for col in all_cols if col not in ['TailNum_vector', 'Origin_vector', 'Dest_vector']])

    # Feature subset selection
    # preprocessing.select_variables(df)

    # # Train/test split
    # train_df, test_df = preprocessing.train_test_split(df, sel_col=all_cols)

    # Tuning
    tuning = Tunning(train_df)
    lr = tuning.run_lr()
    dt = tuning.run_dt()
    rf = tuning.run_rf()


    # Classification
    classifier = LinearRegressionClass()

    if path_testing: # If another dataset was given to test the model on
        classifier.fit(preprocessing.vectorize(df, all_cols)) 
        df_testing = spark.read.csv(path_testing, header=True)
        df_testing = df_testing.limit(100)
        df_testing = preprocessing.prepare_data(df_testing)
        classifier.predict(preprocessing.vectorize(df, all_cols))
    else:
        # Train/test split
        train_df, test_df = preprocessing.train_test_split(df, sel_col=all_cols)
        classifier.fit(train_df)
        classifier.predict(test_df)
