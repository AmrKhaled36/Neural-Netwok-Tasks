import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

scaler = MinMaxScaler()
def normalize_minmax(df, columns):
    """
    Normalize the dataframe using MinMaxScaler.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to normalize.
    columns : list
        The columns to normalize.

    Returns
    -------
    pandas.DataFrame
        The normalized dataframe.
    """

    if isinstance(columns[0], str):
        df[columns] = scaler.fit_transform(df[columns])
    else:
        arr = np.array(columns).reshape(-1,5)
        arr = scaler.transform(arr)
        x1 = arr[0,0]
        x2 = arr[0,1]
        x3 = arr[0,2]
        x4 = arr[0,3]
        x5 = arr[0,4]
        return x1, x2, x3, x4, x5
    return df

def load_x_y(df):
    """
    Load the dataframe into x and y.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to load.
    f1 : str
        The first feature to load.
    f2 : str
        The second feature to load.
    c1 : str
        The first class to load.
    c2 : str
        The second class to load.

    Returns
    -------
    x : numpy.ndarray
        The x data.
    y : numpy.ndarray
        The y data.
    x_train : numpy.ndarray
        The x train data.
    y_train : numpy.ndarray
        The y train data.
    x_test : numpy.ndarray
        The x test data.
    y_test : numpy.ndarray
        The y test data.
    """

    dataframe = df.copy()
    dataframe = normalize_minmax(dataframe, dataframe.columns[0:-1])
    dataframe['bird category'] = dataframe['bird category'].map({"A": 0, "B": 1, "C":2})
    x = dataframe.drop("bird category", axis=1).values
    y = dataframe["bird category"].values

    df_c1 = dataframe[dataframe["bird category"].isin([0])]
    df_c2 = dataframe[dataframe["bird category"].isin([1])]
    df_c3 = dataframe[dataframe["bird category"].isin([2])]

    train_df_c1 = df_c1.sample(n=30)
    train_df_c2 = df_c2.sample(n=30)
    train_df_c3 = df_c3.sample(n=30)

    test_df_c1 = df_c1.drop(train_df_c1.index)
    test_df_c2 = df_c2.drop(train_df_c2.index)
    test_df_c3 = df_c3.drop(train_df_c3.index)

    train_df = pd.concat([train_df_c1, train_df_c2, train_df_c3])
    test_df = pd.concat([test_df_c1, test_df_c2, test_df_c3])

    x_train =  train_df.drop("bird category", axis=1).values
    y_train = train_df["bird category"].values

    x_test = test_df.drop("bird category", axis=1).values
    y_test = test_df["bird category"].values
    
    return x, y, x_train, y_train, x_test, y_test
