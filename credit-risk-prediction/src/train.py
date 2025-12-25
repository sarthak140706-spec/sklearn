import pandas as pd
import joblib
from config import TRAIN_DATA_PATH, TARGET_COLUMN, MODEL_PARAMETERS, MODEL_OUTPUT_DIRECTORY
from data_loader import load_datasets
from preprocessing import build_preprocessing_pipeline, get_columns

# 1. Load training and test data
X_train, y_train, X_test = load_datasets(TRAIN_DATA_PATH, TRAIN_DATA_PATH, TARGET_COLUMN)

# 2. Build preprocessing pipeline and preprocess training data
numeric_cols, categorical_cols = get_columns(pd.concat([X_train, y_train], axis=1), TARGET_COLUMN)
preprocessing_pipeline = build_preprocessing_pipeline(numeric_cols, categorical_cols)
X_train_processed = preprocessing_pipeline.fit_transform(X_train)

# 3. Get model from config
model = MODEL_PARAMETERS['model']

# 4. Train model
model.fit(X_train_processed, y_train)

# 5. Save trained model
model_filename = MODEL_OUTPUT_DIRECTORY / "trained_model.pkl"
joblib.dump(model, model_filename)

# 6. Save preprocessing pipeline
pipeline_filename = MODEL_OUTPUT_DIRECTORY / "preprocessing_pipeline.pkl"
joblib.dump(preprocessing_pipeline, pipeline_filename)

print(f"Model and preprocessing pipeline saved successfully at {MODEL_OUTPUT_DIRECTORY}")
