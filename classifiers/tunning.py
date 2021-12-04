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

		#lr = LogisticRegression(maxIter=10)
		lr = LinearRegression(featuresCol = 'features', labelCol='label')


		print("#########Linear regression default parameters#########")
		print(lr.params)
		print("#########END Linear regression parameters#########")


		paramGrid = ParamGridBuilder() \
    	.addGrid(lr.regParam, [0.01, 0.1,1,10,100]) \
    	.addGrid(lr.elasticNetParam,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]) \
    	.addGrid(lr.maxIter,[10,100,500]) \
    	.build()

		crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		cvModel = crossval.fit(self.df)
		bestModel = cvModel.bestModel

		print("###############Parameters################")
		hyperparams = cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)]
		print(hyperparams)
		print("###############EndParameters#############")

		# print(cvModel.explainParams())

		print("coefficients and intercept")

		# # Print the coefficients and intercept for linear regression
		print("Coefficients: %s" % str(bestModel.coefficients))
		print("Intercept: %s" % str(bestModel.intercept))

		print("Summarize the model over the training set and print out some metrics")
		
		trainingSummary = bestModel.summary
		print("numIterations: %d" % trainingSummary.totalIterations)
		print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
		trainingSummary.residuals.show()
		print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
		print("r2: %f" % trainingSummary.r2)

	def run_dt(self):

		#lr = LogisticRegression(maxIter=10)
		#lr = LinearRegression(featuresCol = 'features', labelCol='label')
		dt = DecisionTreeRegressor(featuresCol='features', labelCol='label')

		print("#########decision tree default parameters#########")
		print(dt.params)
		print("#########END decision tree parameters#########")


		paramGrid = ParamGridBuilder() \
    	.addGrid(dt.minInstancesPerNode, [10,50,100,500]) \
    	.addGrid(dt.maxDepth,[5,10,15,20,25,30]) \
    	.build() #max depth allowed 30

		crossval = CrossValidator(estimator=dt,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		cvModel = crossval.fit(self.df)
		bestModel = cvModel.bestModel

		print("###############Parameters################")
		hyperparams = cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)]
		print(hyperparams)
		print("###############EndParameters#############")

		print("####### Check best parameters #########")
		print(" minInstancesPerNode",bestModel._java_obj.getMinInstancesPerNode())
		print(" max depth",bestModel._java_obj.getMaxDepth())

		print("####### END Check best parameters #########")

		# print(cvModel.explainParams())

		
		print("Summarize the model over the training set and print out some metrics")
		print(bestModel)
		print("End summarize")

	def run_rf(self):
		rf = RandomForestRegressor(featuresCol='features', labelCol='label')

		print("#########random forest  default parameters#########")
		print(rf.params)
		print("#########END random forest parameters#########")


		paramGrid = ParamGridBuilder() \
    	.addGrid(rf.numTrees, [3,5,10]) \
    	.addGrid(rf.maxDepth,[5,10,15,20,25,30]) \
    	.build() #max depth allowed 30

		crossval = CrossValidator(estimator=rf,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		cvModel = crossval.fit(self.df)
		bestModel = cvModel.bestModel

		print("###############Parameters################")
		print("###############Parameters nuevo################")
		print(cvModel.getEstimatorParamMaps())
		print("###############end Parameters nuevo################")
		hyperparams = cvModel.getEstimatorParamMaps()[np.argmax(cvModel.avgMetrics)]
		print(hyperparams)
		print("###############EndParameters#############")

		print("####### Check best parameters #########")
		print(" num of trees",bestModel._java_obj.getNumTrees())
		print(" max depth",bestModel._java_obj.getMaxDepth())

		print("####### END Check best parameters #########")

		# print(cvModel.explainParams())

		
		print("Summarize the model over the training set and print out some metrics")
		print(bestModel)
		print("End summarize")

		print("nuevo best model mapping")
		print(bestModel._java_obj.extractParamMap())
		print("end best model mapping")









