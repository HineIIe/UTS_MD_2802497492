import mlflow
import mlflow.sklearn
import joblib

from classification import run_classification
from regression import run_regression

def run_experiment(df):

    mlflow.set_experiment("Student Placement Pipeline")

    with mlflow.start_run():

        clf_model, clf_metrics = run_classification(df)
        for k, v in clf_metrics.items():
            mlflow.log_metric(f"clf_{k}", v)

        reg_model, reg_metrics = run_regression(df)
        for k, v in reg_metrics.items():
            mlflow.log_metric(f"reg_{k}", v)

        mlflow.sklearn.log_model(clf_model, "classification_model")
        mlflow.sklearn.log_model(reg_model, "regression_model")

        joblib.dump(clf_model, "models/clf_model.pkl")
        joblib.dump(reg_model, "models/reg_model.pkl")