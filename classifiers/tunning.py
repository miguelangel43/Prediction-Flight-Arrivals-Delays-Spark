

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import RegressionEvaluator 
from pyspark.ml.tuning import ParamGridBuilder



    test_object = Tunning()
    test_object.run()

class Tunning:

	def __init__(self):
        self.model = None

    def run(self,data):

    	lr = LogisticRegression(maxIter=10)

		paramGrid = ParamGridBuilder() \
    	.addGrid(lr.regParam, [0.1, 0.01]) \
    	.build()

		crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 

		cvModel = crossval.fit(training)

		print(cvModel.explainParams())







