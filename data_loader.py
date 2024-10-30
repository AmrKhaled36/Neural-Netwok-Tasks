import pandas as pd

def load_x_y(df,f1,f2,c1,c2):
    dataframe = df.copy()
    df = df[[f1,f2, "bird category"]]

    x = df[[f1,f2]].values
    y = df["bird category"].values

    df_c1 = df[df["bird category"].isin([c1])]
    df_c2 = df[df["bird category"].isin([c2])]

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

