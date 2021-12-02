from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline

def train_test_split(df, sel_col=None, perc_train=0.8, perc_test=0.2):
    # Set the variables that will be used
    variables = ['DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'label']
    if sel_col:
        variables = sel_col
    df = df.select(variables)
    # Drop null values
    df = df.na.drop("any")
    # Print some statistics about the variables
    print(df.describe().toPandas().transpose())
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