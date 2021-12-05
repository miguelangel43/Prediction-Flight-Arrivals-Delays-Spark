from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import UnivariateFeatureSelector
from pyspark.sql.functions import col

def vectorize(df, predictor_vars):
    #pred_vars = [e for e in predictor_vars if e != 'label']
    df = df.select(predictor_vars)
    vectorAssembler = VectorAssembler(inputCols = predictor_vars, outputCol = 'features')
    vdf_sel = vectorAssembler.transform(df)
    vdf_sel = vdf_sel.select(['features', 'label'])
    return vdf_sel

def train_test_split(df, sel_col=None, perc_train=0.8, perc_test=0.2):
    # Set the variables that will be used
    variables = ['DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'label']
    if sel_col:
        variables = sel_col
    df = df.select(variables)
    vectorAssembler = VectorAssembler(inputCols = variables, outputCol = 'features')
    vdf_sel = vectorAssembler.transform(df)
    vdf_sel = vdf_sel.select(['features', 'label'])
    train_df, test_df = vdf_sel.randomSplit([perc_train, perc_test])

    return train_df, test_df


def encode_cat_vars(df, cat_column):
    """
    Encodes a categorical variable

    :param df
    :param str cat_column name of the column to be encoded

    :return encoded dataframe with 
    """
    indexer = StringIndexer(inputCol=cat_column, outputCol=cat_column+'_index')
    indexed = indexer.fit(df).transform(df)

    encoder = OneHotEncoder(inputCol=cat_column+'_index',
                        outputCol=cat_column+'_vector')
    model = encoder.fit(indexed)
    encoded = model.transform(indexed)
    encoded = encoded.drop(*(cat_column, cat_column+'_index'))
    encoded = encoded.withColumnRenamed(cat_column+'_vector', cat_column)

    return encoded


def select_variables(df, cols):
    """
    Applies univariate FS on a dataframe

    :params df vector with columns ("features", "label")
    :return df vector with columns ("features", "label", "selectedFeatures")
    """
    vdf_sel = vectorize(df, cols)
    selector = UnivariateFeatureSelector(featuresCol="features", outputCol="selectedFeatures",
                                     labelCol="label", selectionMode="numTopFeatures")
    selector.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(4)

    result = selector.fit(vdf_sel).transform(vdf_sel)
    # print("UnivariateFeatureSelector output with top %d features selected using f_classif"
    #     % selector.getSelectionThreshold())
    #result.show()
    return(result)


def check_data(df):
    countBool = True if df.count()==0 else False


    expected_columns = ['Year', 'Month', 'DayofMonth', 'DayOfWeek', 'DepTime', 'CRSDepTime',
     'ArrTime', 'CRSArrTime', 'UniqueCarrier', 'FlightNum', 'TailNum', 'ActualElapsedTime',
      'CRSElapsedTime', 'AirTime', 'ArrDelay', 'DepDelay', 'Origin', 'Dest', 'Distance', 'TaxiIn', 
      'TaxiOut', 'Cancelled', 'CancellationCode', 'Diverted', 'CarrierDelay', 'WeatherDelay', 'NASDelay', 
      'SecurityDelay', 'LateAircraftDelay']

    received_columns = df.schema.names

    missing_columns = []
    for column in expected_columns:
        if column not in received_columns:
            missing_columns.append(column)

    
    columnsBool = True if len(missing_columns)>0 else False        
    
    if countBool:
        print("The input Dataset does not have any rows")
        exit(1)
    if columnsBool:
        print("The input dataset is missing the following columns",missing_columns)
        exit(1)


def prepare_data(df):
    # Drop the forbidden columns
    forbidden_vars = ('ArrTime', 'ActualElapsedTime', 'AirTime', 'TaxiIn', 'Diverted', 'CarrierDelay',
     'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay')
    df = df.drop(*forbidden_vars)

    # Rename explanatory variable to 'label'
    df = df.withColumnRenamed('ArrDelay', 'label')

    # Drop cancelled flights
    df = df.where("Cancelled == 0")
    # Drop some columns
    drop_cols = ('Cancelled', 'CancellationCode', 'UniqueCarrier', 'TailNum', 'FlightNum')
    df = df.drop(*drop_cols)
    # Drop null values
    df = df.na.drop("any")

    quant_vars = ['Year', 'DayofMonth', 'CRSDepTime', 'CRSElapsedTime', 'TaxiOut',
        'DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'label']
    # Cast to int these quantitative columns
    for var in quant_vars:
        df = df.withColumn(var, col(var).cast("int"))

    # Apply StringIndexer and vectorize the categorical columns
    cat_columns = ["Origin", "Dest", "DayOfWeek", "Month"] # "UniqueCarrier", "CancellationCode", "TailNum"
    for column in cat_columns:
        df = encode_cat_vars(df, column)

    return df#.drop(*set(cat_columns))
