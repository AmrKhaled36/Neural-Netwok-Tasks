import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_minmax(df, columns):
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def load_x_y(df,f1,f2,c1,c2):
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

