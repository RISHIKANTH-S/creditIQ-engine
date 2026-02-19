import pandas as pd
import numpy as np
import joblib


class PredictPipeline:

    def __init__(self):
        self.classifier = joblib.load("artifacts/classifier.pkl")
        self.regressor = joblib.load("artifacts/regressor.pkl")
        self.threshold = joblib.load("artifacts/threshold.pkl")

    def predict(self, applicant_dict):

        data = applicant_dict.copy()

        if data["annual_inc"] <= 0:
            return {"approved": False, "reason": "Invalid income"}

        data["loan_to_income"] = data["loan_amnt"] / data["annual_inc"]
        monthly_income = data["annual_inc"] / 12
        data["dti"] = data["monthly_debt"] / monthly_income

        if data["fico_score"] < 550:
            return {"approved": False, "reason": "Low FICO"}

        if data["dti"] > 0.60:
            return {"approved": False, "reason": "DTI too high"}

        if data["loan_to_income"] > 2.5:
            return {"approved": False, "reason": "Loan-to-income too high"}

        # CLASSIFICATION
        df_input = pd.DataFrame([data])
        prob = self.classifier.predict_proba(df_input)[0][1]

        if prob < self.threshold:
            return {
                "approved": False,
                "approval_probability": round(float(prob), 4),
                "decision": "risk too high"
            }

        # REGRESSION
        data["income_fico"] = data["annual_inc"] * data["fico_score"]
        data["income_term"] = data["annual_inc"] * data["term"]

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

        reg_input = pd.DataFrame([data])[reg_features]

        predicted_log = self.regressor.predict(reg_input)[0]
        predicted_amount = np.expm1(predicted_log)

        max_allowed = 0.35 * data["annual_inc"]
        final_amount = min(predicted_amount, max_allowed)

        return {
            "approved": True,
            "approval_probability": round(float(prob), 4),
            "requested_amount": data["loan_amnt"],
            #"model_predicted_amount": round(float(predicted_amount), 2),
            "Sanctioned Loan Amount": round(float(final_amount), 2)
        }
