import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def load_x_y(df,f1,f2,c1,c2):
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
    dataframe = normalize_minmax(dataframe, [f1,f2])
    dataframe = dataframe[[f1,f2, "bird category"]]
    dataframe['bird category'] = dataframe['bird category'].map({c1: 0, c2: 1})
    x = dataframe[[f1,f2]].values
    y = dataframe["bird category"].values

    df_c1 = dataframe[dataframe["bird category"].isin([0])]
    df_c2 = dataframe[dataframe["bird category"].isin([1])]

    train_df_c1 = df_c1.sample(n=30, random_state=40)
    train_df_c2 = df_c2.sample(n=30, random_state=41)

    test_df_c1 = df_c1.drop(train_df_c1.index)
    test_df_c2 = df_c2.drop(train_df_c2.index)

    train_df = pd.concat([train_df_c1, train_df_c2])
    test_df = pd.concat([test_df_c1, test_df_c2])

    x_train =  train_df[[f1,f2]].values
    y_train = train_df["bird category"].values

    x_test = test_df[[f1,f2]].values
    y_test = test_df["bird category"].values
    
    return x, y, x_train, y_train, x_test, y_test

