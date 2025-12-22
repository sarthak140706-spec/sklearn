Breast Cancer Classification
Project Overview

This project is a Machine Learning pipeline to predict whether a breast tumor is Benign (B) or Malignant (M) using the Breast Cancer Wisconsin Dataset. Implemented in Python with scikit-learn, it includes:

Data preprocessing

Model training and evaluation

Prediction on new samples

Visualization of model performance (Confusion Matrix & ROC Curve)

The trained model achieves high accuracy (~99%) on the test dataset.

Dataset

Source: Breast Cancer Wisconsin (Diagnostic) Dataset

Features: 30 numeric features extracted from images of breast cell nuclei

Target: diagnosis → M (Malignant) or B (Benign)

Total samples: 569

Usage

Full workflow: preprocessing → training → evaluation → prediction

Predict on new samples: provide 30 feature values in the same order as the dataset, then run the prediction script.

Evaluation Metrics

Accuracy: ~99.3%

Precision, Recall, F1-score: Excellent for both classes

Confusion Matrix: Correctly classifies most benign and malignant tumors

ROC-AUC: ~0.99

Visualization includes:

Confusion matrix heatmap

ROC curve

Technologies Used

Python 3.x

Pandas & NumPy (Data handling)

Scikit-learn (Machine Learning & metrics)

Matplotlib & Seaborn (Visualization)

Joblib (Model serialization)

Future Work

Hyperparameter tuning for improved performance

Explore other models (SVM, XGBoost, Logistic Regression)

Deploy the model as a web app for real-time predictions

References

UCI Machine Learning Repository: Breast Cancer Wisconsin Dataset

Scikit-learn Documentation: https://scikit-learn.org/stable/