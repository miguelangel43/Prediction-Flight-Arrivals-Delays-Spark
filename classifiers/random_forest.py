from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

class RandomForestClass:

    def __init__(self):
        self.model = None

    def setModel(self,model):
        self.model = model

    def fit(self, train_df):
        rf = RandomForestRegressor(featuresCol='features', labelCol='label')
        rf_model = rf.fit(train_df)
        self.model = rf_model

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