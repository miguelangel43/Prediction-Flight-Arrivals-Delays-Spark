from pyspark.ml.feature import VectorAssembler

def train_test_split(df, sel_col=None, perc_train=0.8, perc_test=0.2):
    # If the user inputted the variables to use, use those, if not, use all
    variables = ['DepTime', 'DepDelay', 'Distance', 'CRSArrTime', 'ArrDelay']
    if sel_col:
        variables = sel_col
    print(df.describe().toPandas().transpose())
    df = df.select(variables)
    # Print some statistics about the variables
    print(df.describe().toPandas().transpose())
    vectorAssembler = VectorAssembler(inputCols = variables, outputCol = 'features')
    vdf_sel = vectorAssembler.transform(df)
    vdf_sel = vdf_sel.select(['features', 'ArrDelay'])
    train_df, test_df = vdf_sel.randomSplit([perc_train, perc_test])

    return train_df, test_df