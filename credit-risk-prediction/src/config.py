# src/config.py

from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Base & Data Directories
# -------------------------
# Set BASE_DIRECTORY to project root (one level up from src)
BASE_DIRECTORY = Path(__file__).parent.parent
DATA_DIRECTORY = BASE_DIRECTORY / "data"

TRAIN_DATA_PATH = DATA_DIRECTORY / "loan_train.csv"
TEST_DATA_PATH = DATA_DIRECTORY / "loan_test.csv"

# -------------------------
# Target Column
# -------------------------
TARGET_COLUMN = "Loan_Status"

# -------------------------
# Data Split Parameters (optional if you split train/test)
# -------------------------
DATA_SPLIT_PARAMETERS = {
    "test_size": 0.2,
    "random_state": 42,
    "stratify": True
}

# -------------------------
# Preprocessing Parameters
# -------------------------
PREPROCESSING_PARAMETERS = {
    "numerical_scaler": StandardScaler(),  # Scales numeric features
    "categorical_encoder": OneHotEncoder(sparse_output=False, handle_unknown='ignore'),  # Encode categorical
    "missing_value_strategy": "mean"  # Can be "median" or "most_frequent"
}

# -------------------------
# Model Parameters
# -------------------------
MODEL_PARAMETERS = {
    "model": RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )
}

# -------------------------
# Output Directories
# -------------------------
MODEL_OUTPUT_DIRECTORY = BASE_DIRECTORY / "models"
PLOT_OUTPUT_DIRECTORY = BASE_DIRECTORY / "plots"

# Create output directories if they don't exist
MODEL_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
PLOT_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
