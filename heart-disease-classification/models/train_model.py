from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_model(X_train, y_train, model_type):
    if model_type=="RandomForest":
        model = RandomForestClassifier()

    elif model_type=="LogisticRegression":
        model = LogisticRegression()

    elif model_type=="XGBoost":
        model = XGBClassifier()

    else:
        print("Invalid model type")
        return None
    
    model.fit(X_train,y_train)
    print("Training complete")
    print(f"Training accuracy: {model.score(X_train, y_train)}")
    return model