import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


class DataTransformation:

    NUMERIC_FEATURES = [
        "loan_amnt",
        "annual_inc",
        "fico_score",
        "dti",
        "emp_length",
        "term"
    ]

    CATEGORICAL_FEATURES = [
        "purpose",
        "home_ownership"
    ]

    TARGET = "approved"

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df['emp_length'] = (
            df['emp_length']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )

        df['term'] = (
            df['term']
            .astype(str)
            .str.extract(r'(\d+)')
            .astype(float)
        )

        df["loan_to_income"] = (
            df["loan_amnt"]
            .div(df["annual_inc"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

        df.drop(columns=["int_rate"], inplace=True, errors="ignore")

        return df

    def get_preprocessor(self):
        return ColumnTransformer(
            transformers=[
                ("num", "passthrough", self.NUMERIC_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), self.CATEGORICAL_FEATURES)
            ]
        )
