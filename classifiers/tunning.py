

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import LinearRegression





class Tunning:

	def __init__(self,data):
		self.data = data
        

	def run(self):

		#lr = LogisticRegression(maxIter=10)
		lr = LinearRegression(featuresCol = 'features', labelCol='ArrDelay', maxIter=10)

		paramGrid = ParamGridBuilder() \
    	.addGrid(lr.regParam, [0.1, 0.01]) \
    	.build()

		crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		cvModel = crossval.fit(self.data)

		print(cvModel.explainParams())







