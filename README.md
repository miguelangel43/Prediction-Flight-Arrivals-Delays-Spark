# Prediction of Flight Arrival Delays

Application that trains a classifier and predicts flight arrival delays based on past information. Uses the libraries pyspark.ml and pyspark.sql, performs feature engineering, cross-validation and tests various ML algorithms.

## 0. Application Execution Instructions
The data can be run locally the following way with a dataset for training and testing:
spark-submit "<path_application_folder>/app.py" "<path_dataset>"
And the following for a dataset for training and a dataset for testing:
spark-submit <path_application_folder>/app.py <path_train_dataset> <path_test_dataset>

## 1. Introduction
This report explains our project’s pipeline that consists of: data preprocessing, display of some exploratory statistics, hyper-parameter tuning, training and testing of three machine learning models. Moreover, the different decisions made are reasoned and insights from the data and regressors are drawn. The application was run on the dataset dataverse_files/2007.csv.bz2, and as a result, many of the insights about the performance of the models are influenced by this dataset. Nonetheless, based on the information we have about the data, we find a generalization of these insights reasonable.

## 2. Data Preprocessing
At the first preprocessing stage, transformations were carried out to change the data type of the variables (after removing the forbidden ones). As a result the following variables were transformed: 'Year', 'DayofMonth', 'CRSDepTime', 'CRSElapsedTime', 'TaxiOut', 'DepTime', 'DepDelay', 'Distance', 'CRSArrTime', from string to integer. The variables 'Origin', 'Dest', ‘DayofWeek’, ‘Month’ were considered to be categorical and were converted into one-hot vectors. In contrast to ‘DayofWeek’ and ‘Month’, ‘DayofMonth’ and ‘Year’ were not considered to be categorical because they do not present the same level of seasonality and we deemed it more adequate to treat them as quantitative variables. Also, the null values in the dataset were removed.
With the objective of selecting the variables for the application of the prediction model, three steps were carried out. First, an analysis of the dataset was made to manually eliminate variables that were not significant for the prediction. In this first analysis, the variables ‘TailNum’, ‘Cancelled’, ‘FlightNum’, ‘CancellationCode’ and ‘Unique Carrier’ were eliminated since they do not affect the result of the prediction model due to the lack of influence of these variables on the flights’ delay. In step two, a Pearson’s correlation analysis was performed between the remaining variables and the variable ‘ArrDelay’. The variables most correlated with ArrDelay were ‘DepTime’, ‘DepDelay’, ‘CRSArrTime’ , ‘CRSDepTime’ and ‘TaxiOut’ according to their Pearson’s coefficient values, which are the following:

| **Variable**   | **Correlation Values to  ‘ArrDelay’** |
|----------------|---------------------------------------|
| DayofMonth     | 0.016                                 |
| CRSDepTime     | 0.1352                                |
| CRSElapsedTime | 0.0052                                |
| TaxiOut        | 0.325                                 |
| DepTime        | 0.1943                                |
| DepDelay       | 0.9278                                |
| Distance       | -0.0019                               |
| CRSArrTime     | 0.1312                                |
| ArrDelay       | 1.0                                   |

The table indicates that around four variables explain most of the variability of the predicted variable. Therefore, we decided to run Univariate Feature Subset Selection and set the number of variables to select to four. This resulted in the selection of the variables: ‘DepTime’, ‘DepDelay’, ‘CRSArrTime’ and ‘CRSDepTime’.

## 3. Machine Learning Implementation and Validation

MLlib provides a strategy to do model selection and hyperparameter tuning, which consists in the evaluation of a selected model over a grid of the values of the hyperparameters to explore. The evaluation is performed using a CrossValidation technique of fold k = 3 for each possible tuple of hyperparameters on the grid and by selecting the best model as the one with lowest RMSE.
The hyperparameters explored for each model and their corresponding values are the following:
1. **Linear Regression**:
   - Regularization Parameter (regParam): Factor that penalizes the flexibility of the model, that is greater values of the coefficients of the linear regression, which is added to the Residual sum squares minimization. A greater regularization parameter generates a simpler model removing variables with low coefficients. The values explored are : [0.01, 0.1,1,10,100].
  - Elastic Net Parameter (elasticNetParam): Linear regression minimizes the residual sum of squares, and depending on how to penalize the coefficients, could be implemented a L1 (Lasso regularization) or a L2 (Ridge Regularization). The elasticNetParam allows you to use a hybrid version of both regularization, when it’s 0 is a L2 regularization is used and when it’s 1 is a L1 regularization, the coefficient ranges from 0 to 1. The values explored were [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1].
  - Maximum Iterations (maxIter): Maximum amount of iterations used on the minimization. The values explored are [10,100,500].

2. **DecisionTree**:
  - Number of samples per node (minInstancesPerNode): Minimum amount
of samples that has to have at least a node of the tree. The values explored are [10,50,100,500].
  - Maximum depth of tree (maxDepth): Max amount of levels of the tree. The max depth allowed by MLlib is 30. The values explored are [5,10,15,20,25,30].

3. **Random Forest**:
  - Number of Trees (numTrees): The amount of trees to be created. The values explored are [3,5,10].
  - Maximum depth of tree (maxDepth): Maximum amount of levels for each tree to be created. The max depth allowed by MLlib is 30. The values explored are [5,10,15,20,25,30].

