from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pandas.plotting import scatter_matrix
import six
from pyspark.sql import functions as F

class LinearRegressionClassifier:

    def __init__(self):
        self.model = None

    # Input column names as a list of strings
    def fit(self, train_df):
        lr = LinearRegression(featuresCol = 'features', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
        lr_model = lr.fit(train_df)
        trainingSummary = lr_model.summary

        # Print some statistics
        print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
        print("r2: %f" % trainingSummary.r2)
        print("Coefficients: " + str(lr_model.coefficients))
        print("Intercept: " + str(lr_model.intercept))
        train_df.describe().show()
        # Residuals, number of iterations and other statistics
        print("numIterations: %d" % trainingSummary.totalIterations)
        print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
        trainingSummary.residuals.show()

        self.model = lr_model
        
    def predict(self, test_df):
        # Realizamos predicciones sobre el set de prueba y sacamos su R cuadrado
        lr_predictions = self.model.transform(test_df)
        lr_predictions.select("prediction","ArrDelay","features").show(5)
        lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                        labelCol="ArrDelay",metricName="r2")
        print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

        # RMSE 
        test_result = self.model.evaluate(test_df)
        print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)

        # Using the model to make predictions
        predictions = self.model.transform(test_df)
        predictions.select("prediction","ArrDelay","features").show()