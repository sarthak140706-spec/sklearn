import pandas as pd

import os

def load_data(file_path):
    try:
        df =pd.read_csv(file_path,encoding='latin1')

        print("Dataset loaded successfully!")
        print("Number of rows:", df.shape[0])
        print("Number of columns:", df.shape[1])
        print("First 5 rows:\n", df.head())

        return df
    
    except Exception as e:
        print("Error loading dataset:", e)
        return None
    
if __name__ == "__main__":
    # Construct the relative path to the data folder
    base_path = "/workspaces/sklearn/spam_email_classifier/data/spam.csv"
    # Load the dataset
    data = load_data(base_path)