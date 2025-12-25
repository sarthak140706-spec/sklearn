# Credit Risk Prediction

This project predicts the likelihood of loan approval based on applicant data using a **Random Forest Classifier**. It includes data preprocessing, model training, evaluation, and visualization of results.

---

## **Project Structure**

credit-risk-prediction/
│
├─ data/ # Dataset files
│ ├─ loan_train.csv
│ └─ loan_test.csv
│
├─ models/ # Trained model and preprocessing pipeline
│ ├─ trained_model.pkl
│ └─ preprocessing_pipeline.pkl
│
├─ plots/ # All generated plots
│
├─ src/ # Source code
│ ├─ config.py # Configuration (paths, parameters)
│ ├─ data_loader.py # Functions to load datasets
│ ├─ preprocessing.py # Data preprocessing pipelines
│ ├─ train.py # Train the model
│ ├─ evaluate.py # Evaluate the model on test data
│ └─ visualize.py # Generate plots
│
└─ README.md

yaml
Copy code

---

## **Features**

- Handles missing values in numeric and categorical features.
- Scales numeric features and encodes categorical features using a **preprocessing pipeline**.
- Trains a **Random Forest Classifier** with configurable hyperparameters.
- Evaluates model using:
  - Accuracy
  - Confusion matrix
  - Classification report
- Generates visualizations:
  - Target distribution
  - Numeric and categorical feature distributions
  - Correlation heatmap

---

## **Dependencies**

Install required Python packages:

```bash
pip install pandas scikit-learn matplotlib seaborn joblib
Tested with Python 3.10+ and scikit-learn 1.2+

Usage
1. Train the model
bash
Copy code
cd src
python train.py
This will:

Load the training dataset (data/loan_train.csv)

Preprocess the data

Train the Random Forest model

Save the trained model and preprocessing pipeline to models/

2. Evaluate the model
bash
Copy code
cd src
python evaluate.py
This will:

Load the test dataset (data/loan_test.csv)

Preprocess the data

Make predictions

Print evaluation metrics

Save predictions to predictions/y_pred.csv

3. Visualize the data
bash
Copy code
cd src
python visualize.py
Generates plots for:

Target distribution

Numeric feature distributions

Categorical feature distributions

Correlation heatmap

Saves all plots in the plots/ folder.

Configuration
All paths, target column, preprocessing parameters, and model parameters are in src/config.py.

Easily modify:

Preprocessing methods (numerical_scaler, categorical_encoder, missing value strategy)

Model type and hyperparameters

Output directories for models and plots

Notes
Ensure loan_train.csv and loan_test.csv exist in the data/ folder.

Run train.py before evaluate.py.

Visualizations help in understanding feature distributions and target balance.

