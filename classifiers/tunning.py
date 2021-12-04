from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
import numpy as np 

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor

class Tunning:

	def __init__(self,df):
		self.df = df
        

	def run_lr(self):

		# Define Linear Regression Model with basic definitions
		lr = LinearRegression(featuresCol = 'features', labelCol='label')

		# Initialize Grid of hyperparameters to explore
		paramGrid = ParamGridBuilder() \
    	.addGrid(lr.regParam, [0.01, 0.1,1,10,100]) \
    	.addGrid(lr.elasticNetParam,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) \
    	.addGrid(lr.maxIter,[10,100,500]) \
    	.build()

    	# Initialize CrossValidator that will use the model, the grid of hyperparameters and a RegressionEvaluator
		crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		# Fit Model on different scenarios for the Grid
		cvModel = crossval.fit(self.df)

		# Select the best Model
		bestModel = cvModel.bestModel


		print("###############Parameters Linear Regression ################")
		print("-- Tried:")
		print("----- regParam:",[0.01, 0.1,1,10,100])
		print("----- elasticNetParam:",[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
		print("----- maxIter:",[10,100,500])


		# Parameters of the best model
		print("Best Parameters")
		print(bestModel._java_obj.extractParamMap())


		print("coefficients and intercept")

		# Print the coefficients and intercept for linear regression
		print("Coefficients: %s" % str(bestModel.coefficients))
		print("Intercept: %s" % str(bestModel.intercept))

		return bestModel

	def run_dt(self):

		# Define Decission Tree Regresor with basic parameters
		dt = DecisionTreeRegressor(featuresCol='features', labelCol='label')

		# Initialize Grid of hyperparameters to explore
		paramGrid = ParamGridBuilder() \
    	.addGrid(dt.minInstancesPerNode, [10,50,100,500]) \
    	.addGrid(dt.maxDepth,[5,10,15,20,25,30]) \
    	.build() 

    	
    	# Initialize CrossValidator that will use the model, the grid of hyperparameters and a RegressionEvaluator
		crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		# Fit Model on different scenarios for the Grid
		cvModel = crossval.fit(self.df)
		bestModel = cvModel.bestModel

		print("###############Parameters Decision Tree ################")
		print("-- Tried:")
		print("----- minInstancesPerNode:",[10,50,100,500])
		print("----- maxDepth:",[5,10,15,20,25,30])

		
		# Parameters of the best model
		print("Best Parameters for decision tree")
		print(bestModel._java_obj.extractParamMap())

		return bestModel

	def run_rf(self):

		# Define Random Forest with basic parameters
		rf = RandomForestRegressor(featuresCol='features', labelCol='label')

		# Initialize Grid of hyperparameters to explore
		paramGrid = ParamGridBuilder() \
    	.addGrid(rf.numTrees, [3,5,10]) \
    	.addGrid(rf.maxDepth,[5,10,15,20,25,30]) \
    	.build() #max depth allowed 30


    	# Initialize CrossValidator that will use the model, the grid of hyperparameters and a RegressionEvaluator
		crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		cvModel = crossval.fit(self.df)
		bestModel = cvModel.bestModel


		print("###############Parameters Random Forest ################")
		print("-- Tried:")
		print("----- numTres:",[3,5,10])
		print("----- maxDepth:",[5,10,15,20,25,30])


		# Parameters of the best model
		print("Best Parameters for random forest")
		print(bestModel._java_obj.extractParamMap())

		return bestModel
		








