import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from preprocess import X, y
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
import seaborn as sns


model = joblib.load("C:/Users/ASUS/OneDrive/Desktop/breast-cancer-classification/breast_cancer_rf_model.pkl")

y_pred = model.predict(X)
y_pred_proba = model.predict_proba(X)[:,1]

# Accuracy
accuracy = accuracy_score(y, y_pred)
print("Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

# Classification Report
cr = classification_report(y, y_pred)
print("Classification Report:\n", cr)

# ROC-AUC
roc_auc = roc_auc_score(y, y_pred_proba)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Benign', 'Malignant'], yticklabels=['Benign', 'Malignant'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

RocCurveDisplay.from_predictions(y, y_pred_proba)
plt.title("ROC Curve")
plt.show()
