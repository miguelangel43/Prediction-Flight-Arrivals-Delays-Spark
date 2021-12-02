from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

class DecisionTreeClass:

    def __init__(self):
        self.model = None

    def fit(self, train_df):
        dt = DecisionTreeRegressor(featuresCol='features', labelCol='label')
        dt_model = dt.fit(train_df)
        self.model = dt_model

    def predict(self, test_df):
        # Predict the test_df and print stats
        dt_predictions = self.model.transform(test_df)
        dt_predictions.select("prediction","label","features")
        dt_evaluator = RegressionEvaluator(predictionCol="prediction", \
                        labelCol="label",metricName="r2")

        print("R Squared (R2) on test data = %g" % dt_evaluator.evaluate(dt_predictions))