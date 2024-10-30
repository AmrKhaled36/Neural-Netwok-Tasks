import GUI
import tkinter as tk
import pandas as pd

df = pd.read_csv("birds.csv")

print(df.describe())

print(df['gender'].value_counts())

#fill NA with mode
df['gender'] = df['gender'].fillna(df['gender'].mode()[0])

#encode gender column to 1 and 0
df['gender'] = df['gender'].map({'male': 1, 'female': 0})
print(df['gender'].value_counts())
print(df['gender'].unique())




root = tk.Tk()
app = GUI.window(root,df)
root.mainloop()