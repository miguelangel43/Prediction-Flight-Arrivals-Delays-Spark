from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pandas.plotting import scatter_matrix
import six
from pyspark.sql import functions as F

class LinearRegressionClassifier:

    def __init__(self, df):
        self.df = df

    # Input column names as a list of strings
    def fit(self, variables=False, perc_train=0.7, perc_test=0.3):
        # If the user inputted the variables to use, use those, if not, use all
        df = self.df
        if variables:
            sel_col = ['DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'ArrDelay']
            df = df.select(variables)
        # Drop null values
        df = df.na.drop()
        # Print some statistics about the variables
        print(df.describe().toPandas().transpose())

        # Obtenemos los coeficientes de correlación con respecto a la variable ArrDelay
        for i in df.columns:
            if not isinstance(df.select(i).take(1)[0][0], six.string_types):
                print( "Correlation to ArrDelay for ", i, df.stat.corr('ArrDelay',i))
        
        # # Obtenemos los coeficientes de correlación con respecto a la variable ArrDelay
        # for i in df.columns:
        #     if not isinstance(df.select(i).take(1)[0][0], six.string_types):
        #         print( "Correlation to ArrDelay for ", i, df.stat.corr('ArrDelay',i))

        # Preparamos los datos para el aprendizaje automático. Solo necesitamos dos columnas: "features" y ("ArrDelay")
        vectorAssembler = VectorAssembler(inputCols = ['DepTime', 'DepDelay', 'Distance', 'CRSArrTime'], outputCol = 'features')
        vdf_sel = vectorAssembler.transform(df)
        vdf_sel = vdf_sel.select(['features', 'ArrDelay'])
        vdf_sel.show(3)
        
        # Separamos el dataframe en set de entrenamiento y de prueba
        splits = vdf_sel.randomSplit([perc_train, perc_test])
        train_df = splits[0]
        test_df = splits[1]

        # Aplicamos el modelo sobre el set de entrenamiento e imprimimos algunas estadisticas
        lr = LinearRegression(featuresCol = 'features', labelCol='ArrDelay', maxIter=10, regParam=0.3, elasticNetParam=0.8)
        lr_model = lr.fit(train_df)
        trainingSummary = lr_model.summary
        print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
        print("r2: %f" % trainingSummary.r2)
        print("Coefficients: " + str(lr_model.coefficients))
        print("Intercept: " + str(lr_model.intercept))

        # RMSE mide las diferencias entre los valores predichos por el modelo y los valores reales. 
        # Sin embargo, RMSE por sí solo no tiene sentido hasta que lo comparamos con estadisticas reales de "ArrDelay", como la media, el mínimo y el máximo. 
        # Después de tal comparación, nuestro RMSE se ve bastante bien.
        # R cuadrado de 0,86 indica que en nuestro modelo, aproximadamente el 86% de la variabilidad en “ArrDelay” puede explicarse utilizando el modelo.
        train_df.describe().show()

        # Realizamos predicciones sobre el set de prueba y sacamos su R cuadrado
        lr_predictions = lr_model.transform(test_df)
        lr_predictions.select("prediction","ArrDelay","features").show(5)
        
        lr_evaluator = RegressionEvaluator(predictionCol="prediction", \
                        labelCol="ArrDelay",metricName="r2")
        print("R Squared (R2) on test data = %g" % lr_evaluator.evaluate(lr_predictions))

        # RMSE en el set de prueba
        test_result = lr_model.evaluate(test_df)
        print("Root Mean Squared Error (RMSE) on test data = %g" % test_result.rootMeanSquaredError)

        # Residuales, numeros de iteración y otras estadisticas
        print("numIterations: %d" % trainingSummary.totalIterations)
        print("objectiveHistory: %s" % str(trainingSummary.objectiveHistory))
        trainingSummary.residuals.show()

        # Usando nuestro modelo de Regresión Lineal para hacer predicciones
        predictions = lr_model.transform(test_df)
        predictions.select("prediction","ArrDelay","features").show()