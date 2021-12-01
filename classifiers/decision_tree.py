from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

class DecisionTreeClass:

    def __init__(self):
        self.model = None

    def fit(self, train_df):
        dt = DecisionTreeRegressor(featuresCol='features', labelCol='ArrDelay')
        dt_model = dt.fit(train_df)
        self.model = dt_model

    def predict(self, test_df):
        # Predict the test_df and print stats
        dt_predictions = self.model.transform(test_df)
        dt_predictions.select("prediction","ArrDelay","features")
        dt_evaluator = RegressionEvaluator(predictionCol="prediction", \
                        labelCol="ArrDelay",metricName="r2")

        print("R Squared (R2) on test data = %g" % dt_evaluator.evaluate(dt_predictions))