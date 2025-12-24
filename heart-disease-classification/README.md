Heart Disease Classification
Project Overview

This project is a Machine Learning pipeline to predict the presence of heart disease using patient health data. It leverages supervised learning algorithms to classify whether a patient has heart disease based on clinical features. The workflow is modular, including data loading, preprocessing, model training, evaluation, and saving the trained model.

Features

Load and inspect dataset efficiently.

Handle missing values and preprocess features:

Scaling numerical features (MinMaxScaler or StandardScaler).

Encoding categorical variables.

Support for multiple ML models:

Random Forest Classifier

Logistic Regression

XGBoost Classifier

Train-test split with configurable parameters.

Evaluate model performance using:

Accuracy, Precision, Recall, F1-score

Confusion Matrix visualization

Save trained model for future inference.

Installation

Clone the repository:

git clone <repository_url>
cd Heart-Disease-Classification


Create a virtual environment:

python -m venv .venv


Activate the virtual environment:

Windows (PowerShell):

.\.venv\Scripts\Activate


Linux/Mac:

source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt


Required packages include: pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, joblib.

Usage

Set project parameters in config.py:

DATASET_PATH – path to your CSV dataset

TEST_SIZE – train/test split ratio

RANDOM_STATE – reproducibility seed

SCALER_TYPE – "MinMax" or "Standard"

MODEL_SAVE_PATH – path to save trained model

Run the pipeline:

python main.py


Results:

Model metrics printed to console (accuracy, precision, recall, F1-score)

Confusion matrix plotted

Trained model saved at saved_models/heart_disease_model.pkl

Evaluation Metrics

The project evaluates model performance using:

Accuracy – overall correctness of the model.

Precision – proportion of predicted positives that are correct.

Recall – proportion of actual positives that are correctly identified.

F1-score – harmonic mean of precision and recall.

Confusion Matrix – visual representation of true vs predicted labels.

Future Improvements

Hyperparameter tuning using GridSearchCV or RandomizedSearchCV.

Cross-validation to improve model reliability.

Feature importance visualization for better interpretability.

Deploy model as a web app using Flask or FastAPI.