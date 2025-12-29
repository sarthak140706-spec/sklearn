# src/evaluate.py

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from data_loader import load_data
from preprocess import preprocess_data
import os

# Step 1: Load the saved classifier
classifier_path = "/workspaces/sklearn/spam_email_classifier/models/spam_classifier.pkl"
classifier = joblib.load(classifier_path)

# Step 2: Load the saved vectorizer
vectorizer_path = "/workspaces/sklearn/spam_email_classifier/models/vectorizer.pkl"
vectorizer = joblib.load(vectorizer_path)

# Step 3: Load the dataset
dataset_path = "/workspaces/sklearn/spam_email_classifier/data/spam.csv"
data = load_data(dataset_path)

# Step 4: Preprocess the dataset
# This will drop duplicates, encode labels, and split into train/test sets
X_train, X_test, y_train, y_test = preprocess_data(data, text_column="v2", label_column="v1")

# Step 5: Transform test data using the loaded vectorizer
X_test_vectors = vectorizer.transform(X_test)

# Step 6: Make predictions using the trained classifier
y_pred = classifier.predict(X_test_vectors)

# Step 7: Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

print("Evaluation complete!")
