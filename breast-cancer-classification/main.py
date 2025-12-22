# main.py

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay

# ================================
# Step 1: Load and preprocess data
# ================================

df = pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/breast-cancer-classification/data/breast_cancer.csv")

# Drop unnecessary columns
df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')

# Encode target
encoder = LabelEncoder()
target = encoder.fit_transform(df['diagnosis'])

# Separate features and target
X = df.drop('diagnosis', axis=1)
y = target

print("Features shape:", X.shape)
print("Target shape:", y.shape)

# ================================
# Step 2: Split data
# ================================
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# Step 3: Train Random Forest model
# ================================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Save trained model
joblib.dump(model, "breast_cancer_rf_model.pkl")

# ================================
# Step 4: Evaluate the model
# ================================
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:,1]

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# ROC-AUC
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)

# Plot Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Plot ROC Curve
RocCurveDisplay.from_predictions(y_test, y_pred_proba)
plt.title("ROC Curve")
plt.show()

# ================================
# Step 5: Predict a new sample
# ================================
# Example single new sample (replace with actual values)
feature_columns = X.columns.tolist()
sample_values = [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
                  1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193,
                  25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]

new_data = pd.DataFrame(sample_values, columns=feature_columns)

prediction = model.predict(new_data)
label = 'Malignant' if prediction[0] == 1 else 'Benign'
print("Prediction for new sample:", label)
