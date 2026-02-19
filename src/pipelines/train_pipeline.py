import pandas as pd
import joblib

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import logger

def run_training():

    # -------------------------
    # DATA INGESTION
    # -------------------------
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    train_df = pd.read_csv(train_path)

    # -------------------------
    # DATA CLEANING
    # -------------------------
    transformer = DataTransformation()
    train_df = transformer.clean_data(train_df)

    preprocessor = transformer.get_preprocessor()

    features = (
        transformer.NUMERIC_FEATURES +
        transformer.CATEGORICAL_FEATURES
    )

    trainer = ModelTrainer()
    # CLASSIFIER TRAINING
    logger.info("Starting classifier training...")
    classifier, threshold = trainer.train_classifier(
        train_df,
        preprocessor,
        features,
        transformer.TARGET
    )
    logger.info("Classifier training completed")

    logger.info("Starting regressor training")
    regressor = trainer.train_regressor(train_df)
    joblib.dump(classifier, "artifacts/classifier.pkl")
    joblib.dump(threshold, "artifacts/threshold.pkl")
    joblib.dump(regressor, "artifacts/regressor.pkl")
    logger.info("Training completed successfully.")


if __name__ == "__main__":
    run_training()


