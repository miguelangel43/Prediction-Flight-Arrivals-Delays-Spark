
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
from pyspark.ml.functions import vector_to_array

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

    #Getting inputs from console, 1 is path of dataset, 2 (Optional) path to test dataset
    path = sys.argv[1]
    path_testing = False
    if len(sys.argv) == 3:
        path_testing = sys.argv[2]

    # Read the dataset into a spark dataframe
    try:
        df = spark.read.csv(path, header=True)
    except:
        print("The specified path is not pointing a readable file")
        exit(1)
        
    preprocessing.check_data(df)

    # Get only a sample of the rows for faster computation
    df = df.sample(0.00001)

    # Drop columns, cast to adequate datatypes, drop null values and encode categorical variables
    print('Preparing the data...')
    df = preprocessing.prepare_data(df)
    
    print('Dataframe schema')
    df.printSchema()
    
    # All variables that will be considered for the prediction
    all_cols = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSDepTime', 'CRSElapsedTime', 'TaxiOut', 'Origin', 'Dest',
        'DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'label']
    # Categorical variables
    cat_cols = ["Origin", "Dest", "DayOfWeek", "Month"]

    # Print some statistics    
    data_analysis.print_correlations(df, [col for col in all_cols if col not in cat_cols])
    data_analysis.print_stats(df, [col for col in all_cols if col not in cat_cols])


    # Feature subset selection
    fss_data = preprocessing.select_variables(df, all_cols)
    view = (fss_data.withColumn("selectedFeatures", vector_to_array("selectedFeatures"))).select([col("selectedFeatures")[i] for i in range(4)])
    view.show()

    # Tuning
    tuning = Tunning(preprocessing.vectorize(df, all_cols))
    lr = tuning.run_lr()
    dt = tuning.run_dt()
    rf = tuning.run_rf()

    # Classification
    classifier = LinearRegressionClass()

    if path_testing: # If another dataset was given to test the model on
        classifier.fit(preprocessing.vectorize(df, all_cols)) 
        df_testing = spark.read.csv(path_testing, header=True)
        df_testing = preprocessing.prepare_data(df_testing)
        classifier.predict(preprocessing.vectorize(df_testing, all_cols))
    else:
        # Train/test split
        train_df, test_df = preprocessing.train_test_split(df, sel_col=all_cols)
        classifier.fit(train_df)
        classifier.predict(test_df)
