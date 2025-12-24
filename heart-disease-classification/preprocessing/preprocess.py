from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
from data.dataset_loader import load 

def preprocess_data(X, y, test_size, random_state):
    numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    scaler_minmax = MinMaxScaler()
    X[numerical_cols] = scaler_minmax.fit_transform(X[numerical_cols])
    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test