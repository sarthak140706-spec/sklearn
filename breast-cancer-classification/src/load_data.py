import pandas as pd
import numpy as np

df = pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/breast-cancer-classification/data/breast_cancer.csv")
print(df.head(5))
print(df.columns)
print(df.info())
print("Data loaded successfully")