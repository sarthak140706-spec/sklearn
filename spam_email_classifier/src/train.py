from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from preprocess import preprocess_data
from data_loader import load_data
import os

dataset_path = "/workspaces/sklearn/spam_email_classifier/data/spam.csv"
data = load_data(dataset_path)

X_train, X_test, y_train, y_test = preprocess_data(data, text_column="v2", label_column="v1")

vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)

classifier = MultinomialNB()
classifier.fit(X_train_vectors,y_train)

y_pred = classifier.predict(X_test_vectors)

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

# Step 7 (optional): Save the trained model
joblib.dump(classifier, '../models/spam_classifier.pkl')
joblib.dump(vectorizer, '../models/vectorizer.pkl')

print("Training complete. Model saved!")