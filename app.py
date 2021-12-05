
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

def factory_model(model_to_use):
        if model_to_use == 0:
            return LinearRegressionClass()
        if model_to_use == 1:
            return DecisionTreeClass()
        if model_to_use == 2:
            return RandomForestClass()
 
if __name__ == "__main__":

    # Create Spark context with Spark configuration
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("OFF")

    #Getting inputs from console, 1 is path of dataset, 2 (Optional) path to test dataset
    path = sys.argv[1]
    path_testing = False
    print('Reading the data...')
    if len(sys.argv) == 3:
        path_testing = sys.argv[2]

    # Read the dataset into a spark dataframe
    try:
        df = spark.read.csv(path, header=True)
    except:
        print("The specified path is not pointing a readable file")
        exit(1)

    # Check    
    preprocessing.check_data(df)

    # Get only a sample of the rows for faster computation
    #df = df.sample(0.0001)
    #df = df.limit(1000)
    print(df.count())

    # Drop columns, cast to adequate datatypes, drop null values and encode categorical variables
    print('Preparing the data...')
    df = preprocessing.prepare_data(df)
    
    print('Dataframe schema')
    df.printSchema()
    
    # All variables that will be considered for the prediction
    quant_cols = ['Year', 'DayofMonth', 'DepTime', 'CRSDepTime', 'CRSArrTime', 'CRSElapsedTime', 'label',
         'DepDelay', 'Distance', 'TaxiOut']
    # Categorical variables
    cat_cols = ["Origin", "Dest", "DayOfWeek", "Month"]

    # Print some statistics    
    data_analysis.print_stats(df, quant_cols)
    #data_analysis.print_correlations(df, quant_cols)

    # Feature subset selection
    fss_data = preprocessing.select_variables(df, all_cols)
    view = (fss_data.withColumn("selectedFeatures", vector_to_array("selectedFeatures"))).select([col("selectedFeatures")[i] for i in range(4)])
    view.show(5)

    # We continue only with the variables that the FSS selected
    fss_cols = ['DepTime', 'DepDelay', 'CRSArrTime', 'CRSDepTime']
    df = df.select(fss_cols)
    print('Dataframe schema after FSS')
    df.printSchema

    # Classification
    if path_testing: # If another dataset was given to test the model on

        # Tuning
        tuning = Tunning(preprocessing.vectorize(df, df.columns))
        lr = tuning.run_lr()
        dt = tuning.run_dt()
        rf = tuning.run_rf()
        models_to_use = [lr,dt,rf]

        df_testing = spark.read.csv(path_testing, header=True)
        df_testing = preprocessing.prepare_data(df_testing)
        df_testing = df_testing.select(fss_cols)
        vdf_testing = preprocessing.vectorize(df_testing, df_testing.columns)

        print('Testing the tunned models...\n')
        model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']
        for i in range(0,3):
            classifier = factory_model(i)
            classifier.setModel(models_to_use[i])
            print('\n', model_names[i], '\n')
            classifier.predict(vdf_testing)
        
    else:
        # Train/test split
        print('Splitting dataset into train and test...')
        train_df, test_df = preprocessing.train_test_split(df, sel_col=df.columns)

        # Tuning
        tuning = Tunning(train_df)
        print('Tunning a Linear Regression model...')
        lr = tuning.run_lr()
        print('Tunning a Decision Tree model...')
        dt = tuning.run_dt()
        print('Tunning a Random Forest model...')
        rf = tuning.run_rf()
        models_to_use = [lr,dt,rf]

        print('Testing the tunned models...\n')
        model_names = ['Linear Regression', 'Decision Tree', 'Random Forest']
        for i in range(0,3):
            classifier = factory_model(i)
            classifier.setModel(models_to_use[i])
            print('\n', model_names[i], '\n')
            classifier.predict(test_df)
