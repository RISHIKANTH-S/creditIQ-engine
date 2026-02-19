import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    raw_data_path: str = "data/lending_club_80k_clean.csv"
    train_data_path: str = "artifacts/train.csv"
    test_data_path: str = "artifacts/test.csv"


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        df = pd.read_csv(self.config.raw_data_path)

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["approved"]
        )

        train_set.to_csv(self.config.train_data_path, index=False)
        test_set.to_csv(self.config.test_data_path, index=False)

        return (
            self.config.train_data_path,
            self.config.test_data_path
        )
