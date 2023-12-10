# Prediction of Flight Arrival Delays with Spark

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

The hyperparameters chosen for each of the tested models were the following:

| **Model**         | **Parameters Chosen**                             |
|-------------------|---------------------------------------------------|
| Linear Regression | regParam:0.1; elasticNetParam:1; maxIter: 100 |
| Decision Tree     | minInstancesPerNode:10; maxDepth:5            |
| Random Forest     | numTrees:10; maxDepth:15                       |

The optimal parameters obtained for Linear Regression indicate that the regularization used for the optimization is a L1 (Lasso Regularization) since the elasticNetParam is 1, this type of regularization induces that most of the regression coefficients are low except for one of them (due to the shape area allowed by the coefficients[1]), this is exactly what is happening by looking at the values obtained on the coefficients. On the other hand the regularization parameter is almost 0, which means that there is almost no penalization for complex models (big values of the regression coefficients) and this means that even without penalization of this complexity, the model learned is simple indicating that there is a strong linear dependency with the variable that has a non zero coefficient.
Regarding Decision Tree regressor it’s interesting to note that there is low depth on the tree, meaning that the regressor tree does not require to get into detailed levels to preserve the minimization of the RMSE. On the other hand the random forest regressor is optimized when using 10 trees allowing a depth fo 15, which would mean that gets into detailed levels of division of the samples
In order to decide which model to use , the three models have been applied over the test set and in base of a set of metrics the one that performs the best has been chosen as the candidate model, the results are explained in the following section.

## 4. Results and Discussion

In classification, the superiority of a model over the rest can be easily illustrated using metrics like accuracy. This task is more complicated in regression, where the predicted variable is continuous. We mainly used three metrics to determine which was the best model. Those were R Squared, which measures how much variability in the dependent variable can be explained by the model, Mean Square Error (MSE)/Root Mean Square Error (RMSE), which are absolute measures of the fit of the model, and Mean Absolute Error (MAE), which takes the sum of the absolute value of error. In our case, on the dataset we run the models on (800 random samples of the 2007 dataset), linear regression was the best model in all metrics. We therefore conclude that it is the most adequate for prediction. Here are the metrics of the linear regression model:

| Evaluation Metric      | Result |
|-----------|--------------------|
| MSE       | 204.40964663956572 |
| R-Squared | 0.8225314365197474 |
| MAE       | 10.129652649217405 |
| RMSE      | 14.297190165888042 |

With the coefficients [0.0, 0.97349004356602, 0.00018743145306736114, -0.00014491746486146065] and intercept -0.6230026865345778.

## 5. Conclusion
After carrying out this research and based on the results obtained, it is possible to reach the following conclusions:

- Data preprocessing is an essential stage in a data science project. It allows the transformation of the raw data into understandable and usable forms. Raw datasets are usually characterized by incompleteness, inconsistencies, lacking in behavior, and trends while containing errors. In this investigation the preprocessing stage was done in three phases, Business Analysis, Pearson’s Correlation Analysis and Feature Subset Selection, managing to correctly cleaning and transforming the data to be used in the prediction process, the null values were removed and the variables most meaningful were selected .
- It is necessary to highlight that the regularization parameter obtained for the linear regression model is close to 0, meaning that there is almost no penalization to create complex model, but even on this scenario the model is quite simple, assigning coefficients different than 0 to just one variable, so indeed this variable tends to explain a lot the ArrDelay
