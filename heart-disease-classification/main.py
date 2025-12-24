from data.dataset_loader import load
from preprocessing.preprocess import preprocess_data
from models.train_model import train_model
from models.evaluate_model import evaluate_model
from config import DATASET_PATH, TEST_SIZE, RANDOM_STATE, SCALER_TYPE, MODEL_SAVE_PATH
import joblib
import os

# Step 0: Ensure save folder exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# Step 1: Load dataset
X, y = load()

# Step 2: Preprocess data
X_train, X_test, y_train, y_test = preprocess_data(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Step 3: Train model
model = train_model(X_train, y_train, model_type="XGBoost")

# Step 4: Save trained model
joblib.dump(model, MODEL_SAVE_PATH)
print(f"Model saved successfully at: {MODEL_SAVE_PATH}")

# Step 5: Evaluate model
evaluate_model(model, X_test, y_test)
