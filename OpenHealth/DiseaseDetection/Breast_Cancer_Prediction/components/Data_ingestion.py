import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import customexception


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("Artifacts", "Breast_Cancer_Disease", "Raw_data.csv")
    train_data_path: str = os.path.join("Artifacts", "Breast_Cancer_Disease", "Train_data.csv")
    test_data_path: str = os.path.join("Artifacts", "Breast_Cancer_Disease", "Test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Breast Cancer Disease: Data ingestion phase started")
        try:
            # data = pd.read_csv(r"Notebook_Experiments\Data\cancerb.csv")
            data = pd.read_csv(r"Notebook_Experiments\Data\data.csv")
            logging.info("Breast Cancer Disease: Read the data from CSV")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Breast Cancer Disease: Raw data file created")

            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
            logging.info("Breast Cancer Disease: Train/test split completed")

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Breast Cancer Disease: Train and test files created")

            # IMPORTANT: return train first, test second
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.info("Exception occurred while ingesting breast cancer data")
            raise customexception(e, sys)