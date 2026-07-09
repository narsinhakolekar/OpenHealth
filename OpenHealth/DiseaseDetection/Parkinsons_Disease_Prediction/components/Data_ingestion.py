import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import customexception


class DataIngestionConfig:
    raw_data_path = os.path.join("Artifacts", "Parkinsons_Disease", "Raw_data.csv")
    train_data_path = os.path.join("Artifacts", "Parkinsons_Disease", "Train_data.csv")
    test_data_path = os.path.join("Artifacts", "Parkinsons_Disease", "Test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Parkinson's Disease: Data ingestion started")
        try:
            data = pd.read_csv(r"Notebook_Experiments\Data\Parkinsson disease.csv")
            logging.info("Parkinson's Disease: Dataset loaded successfully")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            data.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Parkinson's Disease: Raw data saved")

            train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Parkinson's Disease: Train/Test split completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.info("Exception occurred in Data Ingestion")
            raise customexception(e, sys)