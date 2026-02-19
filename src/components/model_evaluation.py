import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    r2_score
)


class ModelEvaluation:

    def evaluate_classifier(self, model, X_test, y_test, threshold):

        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)

        metrics = {
            "threshold": float(threshold),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred))
        }

        return metrics


    def evaluate_regressor(self, model, X_test, y_test):

        y_pred_log = model.predict(X_test)

        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_test)

        metrics = {
            "mae": float(mean_absolute_error(y_actual, y_pred)),
            "r2": float(r2_score(y_actual, y_pred))
        }

        return metrics

