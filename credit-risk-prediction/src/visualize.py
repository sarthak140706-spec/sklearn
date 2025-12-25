# src/visualize.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import TRAIN_DATA_PATH, TARGET_COLUMN, PLOT_OUTPUT_DIRECTORY
from data_loader import load_training_data

# -------------------------
# Load training data
# -------------------------
X_train, y_train = load_training_data(TRAIN_DATA_PATH, TARGET_COLUMN)
df_train = pd.concat([X_train, y_train], axis=1)

# -------------------------
# Set seaborn style
# -------------------------
sns.set(style="whitegrid")

# -------------------------
# 1. Target Distribution
# -------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x=TARGET_COLUMN, data=df_train)
plt.title("Target Distribution")
plt.savefig(PLOT_OUTPUT_DIRECTORY / "target_distribution.png")
plt.close()

# -------------------------
# 2. Numeric Feature Distributions
# -------------------------
numeric_cols = df_train.select_dtypes(include='number').columns.tolist()

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df_train[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}")
    plt.savefig(PLOT_OUTPUT_DIRECTORY / f"{col}_distribution.png")
    plt.close()

# -------------------------
# 3. Categorical Feature Distributions
# -------------------------
categorical_cols = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
categorical_cols = [col for col in categorical_cols if col != TARGET_COLUMN]

for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df_train)
    plt.title(f"Distribution of {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_DIRECTORY / f"{col}_distribution.png")
    plt.close()

# -------------------------
# 4. Correlation Heatmap (numeric features only)
# -------------------------
if len(numeric_cols) > 1:
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_train[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.savefig(PLOT_OUTPUT_DIRECTORY / "correlation_heatmap.png")
    plt.close()

print(f"All plots saved successfully in {PLOT_OUTPUT_DIRECTORY}")
