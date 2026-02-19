import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier, LGBMRegressor


class ModelTrainer:

    def train_classifier(self, train_df, preprocessor, features, target):

        X = train_df[features]
        y = train_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42
        )

        lgbm = LGBMClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight={0: 2, 1: 1}
        )

        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", lgbm)
        ])

        param_grid = {
            "classifier__n_estimators": [300, 500],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__num_leaves": [31, 63]
        }

        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            scoring="precision",
            cv=5,
            n_jobs=1,
            verbose=2
        )

        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Threshold optimization
        probs = best_model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, probs)

        precision = precision[:-1]
        recall = recall[:-1]

        valid_idx = np.where(precision >= 0.90)[0]
        best_idx = valid_idx[np.argmax(recall[valid_idx])]
        optimized_threshold = thresholds[best_idx]

        return best_model, optimized_threshold
    

    def train_regressor(self, train_df):
        df_reg = train_df[train_df["approved"] == 1].copy()

        if df_reg.empty:
            raise ValueError("No approved loans available for regression")

        df_reg["income_fico"] = df_reg["annual_inc"] * df_reg["fico_score"]
        df_reg["income_term"] = df_reg["annual_inc"] * df_reg["term"]

        reg_numeric_features = [
            "annual_inc",
            "fico_score",
            "dti",
            "term",
            "emp_length",
            "income_fico",
            "income_term"
        ]

        reg_categorical_features = [
            "purpose",
            "home_ownership"
        ]

        reg_features = reg_numeric_features + reg_categorical_features

        X = df_reg[reg_features]
        y = np.log1p(df_reg["loan_amnt"])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        reg_preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", reg_numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), reg_categorical_features)
            ]
        )

        regressor = LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.03,
            num_leaves=64,
            max_depth=-1,
            random_state=42
        )

        pipeline = Pipeline([
            ("preprocess", reg_preprocessor),
            ("regressor", regressor)
        ])

        pipeline.fit(X_train, y_train)

        return pipeline

