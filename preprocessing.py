from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.feature import UnivariateFeatureSelector

def vectorize(df, predictor_vars):
    pred_vars = [e for e in predictor_vars if e != 'label']
    vectorAssembler = VectorAssembler(inputCols = pred_vars, outputCol = 'features')
    vdf_sel = vectorAssembler.transform(df)
    vdf_sel = vdf_sel.select(['features', 'label'])
    return vdf_sel

def train_test_split(df, sel_col=None, perc_train=0.8, perc_test=0.2):
    # Set the variables that will be used
    variables = ['DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'label']
    if sel_col:
        variables = sel_col
    df = df.select(variables)
    # Drop null values
    df = df.na.drop("any")
    # Print some statistics about the variables
    #print(df.describe().toPandas().transpose())
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
    return encoded


def select_variables(df):
    """
    Applies univariate FS on a dataframe

    :params df vector with columns ("features", "label")
    :return df vector with columns ("features", "label", "selectedFeatures")
    """
    vdf_sel = vectorize(df, df.columns)
    selector = UnivariateFeatureSelector(featuresCol="features", outputCol="selectedFeatures",
                                     labelCol="label", selectionMode="numTopFeatures")
    selector.setFeatureType("continuous").setLabelType("continuous").setSelectionThreshold(4)

    result = selector.fit(vdf_sel).transform(vdf_sel)
    # print("UnivariateFeatureSelector output with top %d features selected using f_classif"
    #     % selector.getSelectionThreshold())
    #result.show()
    return(result)