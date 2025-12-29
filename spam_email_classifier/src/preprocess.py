import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_loader import load_data
import os

def preprocess_data(df, text_column, label_column):
    try:
        # 1. Drop duplicates
        df = df.drop_duplicates()

        # 2. Drop rows with missing values in important columns
        df = df.dropna(subset=[text_column, label_column])

        # 3. Encode labels (spam/ham -> 1/0)
        le = LabelEncoder()
        df[label_column] = le.fit_transform(df[label_column])

        # 4. Split into features and labels
        X = df[text_column]
        y = df[label_column]

        # 5. Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Preprocessing complete!")
        print("Training samples:", X_train.shape[0])
        print("Testing samples:", X_test.shape[0])

        return X_train, X_test, y_train, y_test

    except Exception as e:
        print("Error during preprocessing:", e)
        return None, None, None, None

if __name__ == "__main__":
    # Load dataset
    base_path = "/workspaces/sklearn/spam_email_classifier/data/spam.csv"
    data = load_data(base_path)

    # Preprocess using correct column names
    X_train, X_test, y_train, y_test = preprocess_data(
        data, text_column="v2", label_column="v1"
    )
