from pyspark.ml.stat import Summarizer
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
import six
import preprocessing
import pandas as pd


def print_correlations(df, cols):
    """
    Prints correlation of the predictor variables against the explanatory variable.

    :param df
    :param cols list of column names to calculate correlation with
    """
    for i in cols:
        if not( isinstance(df.select(i).take(1)[0][0], six.string_types)):
            print( "Correlation to ArrDelay for ", i, df.stat.corr('label',i))


def print_stats(df,rows):
    """
    Prints stats of the dataframe's columns

    :params df
    :params rows list of strings of var names
    """
    vdf_sel = preprocessing.vectorize(df, rows)
    summarizer = Summarizer.metrics("mean", "count", "max", "min","std", "sum", "variance")
    means = vdf_sel.select(summarizer.summary(vdf_sel.features))
    means_list = list(means.toPandas().iloc[0,0][0])
    counts_list = means.toPandas().iloc[0,0][1]
    max_list = list(means.toPandas().iloc[0,0][2])
    min_list = list(means.toPandas().iloc[0,0][3])
    std_list = list(means.toPandas().iloc[0,0][4])
    sum_list = list(means.toPandas().iloc[0,0][5])
    var_list = list(means.toPandas().iloc[0,0][6])
    
    stats = pd.DataFrame(list(zip(rows,means_list,max_list,min_list,std_list,var_list)), columns = ['Variable','Mean','Max','Min', 'STD', 'Var']).set_index('Variable')
    print(stats)