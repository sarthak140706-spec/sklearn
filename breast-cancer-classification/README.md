🩺 Breast Cancer Classification

🚀 Live Demo

👉 Try the deployed application here:
🔗 https://sarthak140706-spec-sklea-breast-cancer-classificationapp-h1tdse.streamlit.app/

📌 Project Overview

This project is a complete Machine Learning pipeline to predict whether a breast tumor is Benign (B) or Malignant (M) using the Breast Cancer Wisconsin Dataset.

It includes:

Data preprocessing
Model training using Random Forest
Model evaluation
Prediction on new samples
Interactive Streamlit web app for deployment

The model achieves ~99% accuracy on test data.

📊 Dataset
Source: Breast Cancer Wisconsin (Diagnostic) Dataset
Samples: 569
Features: 30 numeric features derived from cell nucleus images
Target:
M → Malignant
B → Benign
⚙️ Workflow
Data Cleaning & Preprocessing
Feature Selection
Model Training (Random Forest)
Evaluation
Deployment using Streamlit
📈 Evaluation Metrics
Accuracy: ~99.3%
Precision / Recall / F1-score: High performance across both classes
ROC-AUC: ~0.99
Visualizations:
Confusion Matrix Heatmap
ROC Curve
🧠 Model Details
Algorithm: Random Forest Classifier
Input Features: 30 numeric features
Output: Binary classification (Benign / Malignant)
Serialization: Joblib
🛠️ Tech Stack
Python 🐍
Pandas & NumPy 📊
Scikit-learn 🤖
Matplotlib & Seaborn 📉
Streamlit 🌐
Joblib 💾
Plotly 📈
🚀 Usage
Run Locally
git clone https://github.com/sarthak140706-spec/sklearn.git
cd sklearn/breast-cancer-classification
pip install -r requirements.txt
streamlit run app.py
Prediction Input

Provide 30 feature values in the same order as the dataset to get predictions.

🔮 Future Improvements
Hyperparameter tuning for better generalization
Experiment with SVM, XGBoost, Logistic Regression
Add SHAP explainability for model interpretability
Improve UI with advanced medical dashboard components
Mobile-friendly deployment
📚 References
UCI Machine Learning Repository: Breast Cancer Dataset
https://scikit-learn.org/stable/
⚠️ Disclaimer

This project is for educational purposes only and should not be used for real medical diagnosis.