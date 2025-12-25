import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from config import TEST_DATA_PATH, TARGET_COLUMN, MODEL_OUTPUT_DIRECTORY
from data_loader import load_test_data
from pathlib import Path

# 1. Load trained model and preprocessing pipeline
model = joblib.load(MODEL_OUTPUT_DIRECTORY / "trained_model.pkl")
preprocessing_pipeline = joblib.load(MODEL_OUTPUT_DIRECTORY / "preprocessing_pipeline.pkl")

# 2. Load test data
df_test = load_test_data(TEST_DATA_PATH)

# 3. Separate features and target
if TARGET_COLUMN in df_test.columns:
    X_test = df_test.drop(columns=[TARGET_COLUMN])
    y_test = df_test[TARGET_COLUMN]
else:
    X_test = df_test.copy()
    y_test = None

# 4. Preprocess test data
X_test_processed = preprocessing_pipeline.transform(X_test)

# 5. Make predictions
y_pred = model.predict(X_test_processed)

# 6. Evaluate model
if y_test is not None:
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)

# 7. Save predictions
predictions_dir = Path("predictions")
predictions_dir.mkdir(parents=True, exist_ok=True)
pred_file = predictions_dir / "y_pred.csv"

if y_test is not None:
    output_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
else:
    output_df = pd.DataFrame({"Predicted": y_pred})

output_df.to_csv(pred_file, index=False)
print(f"Predictions saved successfully at {pred_file}")
