import pandas as pd
import joblib
import numpy as np
import json

from src.components.data_transformation import DataTransformation
from src.components.model_evaluation import ModelEvaluation


def run_evaluation():

    # Load artifacts
    classifier = joblib.load("artifacts/classifier.pkl")
    regressor = joblib.load("artifacts/regressor.pkl")
    threshold = joblib.load("artifacts/threshold.pkl")

    # Load test data
    df = pd.read_csv("artifacts/test.csv")

    transformer = DataTransformation()
    df = transformer.clean_data(df)

    features = (
        transformer.NUMERIC_FEATURES +
        transformer.CATEGORICAL_FEATURES
    )

    X = df[features]
    y = df["approved"]

    evaluator = ModelEvaluation()

    # -------- CLASSIFIER --------
    classifier_metrics = evaluator.evaluate_classifier(
        classifier, X, y, threshold
    )

    # -------- REGRESSOR --------
    df_reg = df[df["approved"] == 1].copy()

    df_reg["income_fico"] = df_reg["annual_inc"] * df_reg["fico_score"]
    df_reg["income_term"] = df_reg["annual_inc"] * df_reg["term"]

    reg_features = [
        "annual_inc",
        "fico_score",
        "dti",
        "term",
        "emp_length",
        "purpose",
        "home_ownership",
        "income_fico",
        "income_term"
    ]

    X_reg = df_reg[reg_features]
    y_reg = np.log1p(df_reg["loan_amnt"])

    regressor_metrics = evaluator.evaluate_regressor(
        regressor, X_reg, y_reg
    )

    # Combine metrics
    all_metrics = {
        "classifier": classifier_metrics,
        "regressor": regressor_metrics
    }

    # Save to artifacts
    with open("artifacts/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=4)

    print("Evaluation metrics saved to artifacts/metrics.json")


if __name__ == "__main__":
    run_evaluation()
