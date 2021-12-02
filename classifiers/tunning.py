from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline


class Tunning:

	def __init__(self,df):
		self.df = df
        

	def run(self):

		#lr = LogisticRegression(maxIter=10)
		lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)


		paramGrid = ParamGridBuilder() \
    	.addGrid(lr.regParam, [0.1, 0.01]) \
    	.build()

		crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		cvModel = crossval.fit(self.df)

		print(cvModel.explainParams())







