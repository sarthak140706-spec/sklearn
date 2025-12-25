import pandas as pd

def load_training_data(dataset_path, target_column):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' missing in training data")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def load_test_data(dataset_path):
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    return df

def load_datasets(train_path, test_path, target_column):
    X_train, y_train = load_training_data(train_path, target_column)
    X_test = load_test_data(test_path)
    return X_train, y_train, X_test
