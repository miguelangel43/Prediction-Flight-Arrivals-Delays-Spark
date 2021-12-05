from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

class LinearRegressionClass:

    def __init__(self):
        self.model = None

    def setModel(self,model):
        self.model = model

    def fit(self, train_df):
        lr = LinearRegression(featuresCol = 'features', labelCol='label', maxIter=10, regParam=0.3, elasticNetParam=0.8)
        lr_model = lr.fit(train_df)

        summary = lr_model.summary
        print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
        print("T Values: " + str(summary.tValues))
        print("P Values: " + str(summary.pValues))
        print("Dispersion: " + str(summary.dispersion))
        print("Null Deviance: " + str(summary.nullDeviance))
        print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
        print("Deviance: " + str(summary.deviance))
        print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
        print("AIC: " + str(summary.aic))
        print("Deviance Residuals: ")
        summary.residuals().show()

        self.model = lr_model
        
    def predict(self, test_df):
        # Predict the test_df and print stats
        lr_predictions = self.model.transform(test_df)
        lr_predictions.select("prediction","label","features")
        lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                        labelCol="label",metricName="r2")

        print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))
        test_result = self.model.evaluate(test_df)
        print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)


