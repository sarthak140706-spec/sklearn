import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import PREPROCESSING_PARAMETERS

def get_columns(df, target_column):
    numeric_columns = df.select_dtypes(include='number').columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if target_column in numeric_columns:
        numeric_columns.remove(target_column)
    if target_column in categorical_columns:
        categorical_columns.remove(target_column)
    
    return numeric_columns, categorical_columns

def build_preprocessing_pipeline(numeric_columns, categorical_columns):
    # Numeric pipeline
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=PREPROCESSING_PARAMETERS['missing_value_strategy'])),
        ('scaler', PREPROCESSING_PARAMETERS['numerical_scaler'])
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', PREPROCESSING_PARAMETERS['categorical_encoder'])
    ])
    
    # Combine pipelines
    preprocessing_pipeline = ColumnTransformer(transformers=[
        ('num', numeric_pipeline, numeric_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])
    
    return preprocessing_pipeline

def preprocess(df, target_column):
    numeric_cols, categorical_cols = get_columns(df, target_column)
    pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)
    
    X = df.drop(columns=[target_column])
    y = df[target_column] if target_column in df.columns else None
    
    X_processed = pipeline.fit_transform(X)
    return X_processed, y
