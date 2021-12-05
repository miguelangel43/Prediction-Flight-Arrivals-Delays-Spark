from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator

class DecisionTreeClass:

    def __init__(self):
        self.model = None

    def setModel(self,model):
        self.model = model

    def fit(self, train_df):
        dt = DecisionTreeRegressor(featuresCol='features', labelCol='label')
        dt_model = dt.fit(train_df)
        self.model = dt_model

    def predict(self, test_df):
        # Predict the test_df and print stats
        predictions = self.model.transform(test_df)
        metrics = ['rmse', 'mse', 'r2', 'mae', 'var']
        metrics_names = ['Root Mean Squared Error', 'Mean Squared Error', 'R Squared (R2)',
            'mean absolute error', 'explained variance']
        evaluator = RegressionEvaluator(predictionCol="prediction", \
                        labelCol="label")
        for i in range(len(metrics)):            
            print(metrics_names[i], "on test data =", evaluator.evaluate(predictions, {evaluator.metricName: metrics[i]}))