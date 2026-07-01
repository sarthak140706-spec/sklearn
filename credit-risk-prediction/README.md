# 💳 Credit Risk Prediction

A complete **Machine Learning** project that predicts whether a loan application is likely to be approved based on applicant information. The project includes **data preprocessing, model training, evaluation, visualization, and a deployed Streamlit web application**.

## 🌐 Live Demo

**Render:** https://sklearn-1.onrender.com/

---

# 📌 Features

* Interactive Streamlit web application
* Predicts loan approval in real time
* Displays prediction confidence
* Complete preprocessing pipeline
* Random Forest Classifier model
* Handles missing values automatically
* Feature scaling and categorical encoding
* Model evaluation with multiple metrics
* Data visualization and exploratory analysis
* Saved preprocessing pipeline and trained model for inference

---

# 📂 Project Structure

```text
credit-risk-prediction/
│
├── app.py                          # Streamlit application
├── main.py                         # Project entry point
├── requirements.txt
├── runtime.txt
├── README.md
│
├── data/
│   ├── loan_train.csv
│   └── loan_test.csv
│
├── models/
│   ├── trained_model.pkl
│   └── preprocessing_pipeline.pkl
│
├── plots/
│   ├── correlation_heatmap.png
│   ├── target_distribution.png
│   └── ...
│
└── src/
    ├── config.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
    ├── visualize.py
    └── predictions/
        └── y_pred.csv
```

---

# 🛠 Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Matplotlib
* Seaborn

---

# 🤖 Machine Learning Pipeline

### Data Preprocessing

* Missing value imputation
* Feature scaling
* One-hot encoding of categorical variables
* Transformation pipeline using Scikit-learn

### Model

* Random Forest Classifier

### Evaluation Metrics

* Accuracy Score
* Classification Report
* Confusion Matrix
* Prediction Probability

---

# 🚀 Running the Project Locally

## Clone the repository

```bash
git clone https://github.com/sarthak140706-spec/sklearn.git
cd sklearn/credit-risk-prediction
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Launch the Streamlit application

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

# 📊 Model Training

Run the training script:

```bash
cd src
python train.py
```

This will:

* Load the training dataset
* Perform preprocessing
* Train the Random Forest model
* Save:

  * `trained_model.pkl`
  * `preprocessing_pipeline.pkl`

---

# 📈 Model Evaluation

```bash
cd src
python evaluate.py
```

Outputs:

* Accuracy
* Classification Report
* Confusion Matrix
* Predictions saved to:

```text
src/predictions/y_pred.csv
```

---

# 📉 Data Visualization

Generate exploratory plots:

```bash
cd src
python visualize.py
```

Generated visualizations include:

* Target distribution
* Correlation heatmap
* Numerical feature distributions
* Categorical feature distributions

All plots are stored in the **plots/** directory.

---

# 📋 Input Features

The application predicts loan approval using:

* Gender
* Married
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

---

# 📷 Application Preview

> Add screenshots of the Streamlit application here for a better GitHub presentation.

---

# 🌍 Live Application

**Render Deployment**

https://sklearn-1.onrender.com/

---

# 👨‍💻 Author

**Sarthak Jadhav**

B.Tech – Artificial Intelligence & Data Science

Passionate about Machine Learning, Deep Learning, Python, and AI-powered applications.

---

## ⭐ If you found this project useful, consider giving the repository a star!
