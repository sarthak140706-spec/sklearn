import joblib
import os

classifier_path = "/workspaces/sklearn/spam_email_classifier/models/spam_classifier.pkl"
classifier = joblib.load(classifier_path)

vectorizer_path = "/workspaces/sklearn/spam_email_classifier/models/vectorizer.pkl"
vectorizer = joblib.load(vectorizer_path)

def predict_email(text):
    text_vector = vectorizer.transform([text])
    
    prediction = classifier.predict(text_vector)
    
    if prediction[0] == 1:
        return "spam"
    else:
        return "ham"

if __name__ == "__main__":
    new_email = "Congratulations! You have won a free lottery. Click here to claim."
    
    result = predict_email(new_email)
    print("The email is classified as:", result)

    # Optional: predict multiple emails
    # emails = ["Free tickets waiting for you!", "Hello, how are you?"]
    # for email in emails:
    #     print(email, "->", predict_email(email))
