import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/breast-cancer-classification/data/breast_cancer.csv")

# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Encode target
encoder = LabelEncoder()
target = encoder.fit_transform(df['diagnosis'])

# Separate features and target
X = df.drop('diagnosis', axis=1)
y = target

# Optional: print shapes
print("Features shape:", X.shape)
print("Target shape:", y.shape)
