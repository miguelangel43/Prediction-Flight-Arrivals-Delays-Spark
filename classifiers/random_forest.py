from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

class RandomForestClass:

    def __init__(self):
        self.model = None

    def fit(self, train_df):
        rf = RandomForestRegressor(featuresCol='features', labelCol='label')
        rf_model = rf.fit(train_df)
        self.model = rf_model

    def predict(self, test_df):
        # Predict the test_df and print stats
        rf_predictions = self.model.transform(test_df)
        rf_predictions.select("prediction","label","features")
        rf_evaluator = RegressionEvaluator(predictionCol="prediction", \
                        labelCol="label",metricName="r2")

        print("R Squared (R2) on test data = %g" % rf_evaluator.evaluate(rf_predictions))