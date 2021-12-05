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
        predictions = self.model.transform(test_df)
        #predictions.select("prediction","label","features")
        
        metrics = ['rmse', 'mse', 'r2', 'mae', 'var']
        metrics_names = ['Root Mean Squared Error', 'Mean Squared Error', 'R Squared (R2)',
            'mean absolute error', 'explained variance']

        evaluator = RegressionEvaluator(predictionCol="prediction", \
                        labelCol="label")
        for i in range(len(metrics)):            
            print(metrics_names[i], "on test data =", evaluator.evaluate(predictions, {evaluator.metricName: metrics[i]}))
