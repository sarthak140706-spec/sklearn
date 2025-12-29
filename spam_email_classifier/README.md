Spam Email Classifier

A Machine Learning project to classify emails as spam or ham (not spam) using Python, scikit-learn, and Naive Bayes. The project includes data preprocessing, model training, evaluation, and prediction on new emails.

Project Structure
spam_email_classifier/
│
├── data/
│   └── spam.csv                   # Dataset containing emails and labels
│
├── models/
│   ├── spam_classifier.pkl        # Trained Naive Bayes model
│   └── vectorizer.pkl             # Saved text vectorizer
│
├── src/
│   ├── data_loader.py             # Load dataset
│   ├── preprocess.py              # Preprocess and split data
│   ├── train.py                   # Train the Naive Bayes classifier
│   ├── evaluate.py                # Evaluate the trained model
│   └── predict.py                 # Predict new emails
│
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation

Dataset

The dataset (spam.csv) contains emails and their labels (spam or ham).

Example columns:

v1	v2
ham	"Go until jurong point, crazy..."
spam	"Free entry in 2 a wkly comp to win FA Cup..."

v1 → Label (spam or ham)

v2 → Email text

Setup & Installation

Clone the repository:

git clone <repo_url>
cd spam_email_classifier


Install dependencies:

pip install -r requirements.txt


Run in GitHub Codespaces:
Open the repository in Codespaces and open a terminal inside /src folder.

Usage
1. Load and preprocess data
python data_loader.py
python preprocess.py


data_loader.py → Loads the CSV dataset.

preprocess.py → Cleans data, encodes labels, and splits into train/test sets.

2. Train the classifier
python train.py


Trains a Multinomial Naive Bayes classifier.

Converts emails to numeric vectors using CountVectorizer.

Saves the trained model (spam_classifier.pkl) and vectorizer (vectorizer.pkl) in models/.

3. Evaluate the model
python evaluate.py


Evaluates the trained model on the test set.

Outputs accuracy, confusion matrix, and classification report.

4. Predict new emails
python predict.py


Classifies a new email as spam or ham.

Can be modified to predict multiple emails at once.

Requirements

requirements.txt should include:

pandas
scikit-learn
joblib


Install using:

pip install -r requirements.txt

Key Features

Text preprocessing (duplicates removal, missing values handling)

Label encoding (spam → 1, ham → 0)

Vectorization of email text (CountVectorizer)

Multinomial Naive Bayes classification

Evaluation metrics: Accuracy, Confusion Matrix, Precision, Recall, F1-Score

Save & load trained model and vectorizer for predictions

Future Improvements

Use TF-IDF vectorization instead of CountVectorizer.

Try other classifiers like Logistic Regression or Random Forest.

Implement a GUI or web interface for real-time spam detection.

Add email preprocessing (removing links, punctuation, stopwords) for better accuracy.
