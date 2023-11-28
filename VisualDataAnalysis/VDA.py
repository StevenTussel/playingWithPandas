# Matplotlib forms basis for visualization in Python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
sns.set()
import os


DATA_PATH = "https://raw.githubusercontent.com/Yorko/mlcourse.ai/main/data/"
df = pd.read_csv(DATA_PATH + "mlbootcamp5_train.csv", sep=";")
print(df.head())

#Question 1.2. (1 point). Who more often report consuming alcohol – men or women?
print(df.groupby('gender')["alco"].mean())


#Question 1.3. (1 point). What’s the rounded difference between the percentages of smokers among men and women?
print(df.groupby('gender')['smoke'].mean())


#Question 1.4. (1 point). What’s the rounded difference between median values of age (in months)
#  for non-smokers and smokers? You’ll need to figure out the units of feature age in this dataset.
print(df.groupby('smoke')["age"].median() / 365.25)

#Create a new feature – BMI (Body Mass Index). 
# To do this, divide weight in kilograms by the square of height in meters. Normal BMI values are said to be from 18.5 to 25.
df["BMI"] = df["weight"]/ (df["height"] / 100 ) **2
print(df['BMI'].median())

df_to_remove = df[
    (df["ap_lo"] > df["ap_hi"])
    | (df["height"] < df["height"].quantile(0.025))
    | (df["height"] > df["height"].quantile(0.975))
    | (df["weight"] < df["weight"].quantile(0.025))
    | (df["weight"] > df["weight"].quantile(0.975))
]
print(df_to_remove.shape[0] / df.shape[0])

filtered_df = df[~df.index.isin(df_to_remove)]