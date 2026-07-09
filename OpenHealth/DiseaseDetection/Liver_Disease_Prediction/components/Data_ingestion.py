import os
import sys
import pandas as pd
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("Artifacts", "Liver_Disease", "liver_raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # original CSV path you gave
            source_csv = r"C:\Users\narsi\Desktop\Coding\projects\OpenHealth\Notebook_Experiments\Data\indian_liver_patient.csv"

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            df = pd.read_csv(source_csv)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            print("Liver data ingestion completed.")
            print("Saved raw liver data to:", self.ingestion_config.raw_data_path)

            return self.ingestion_config.raw_data_path

        except Exception as e:
            raise Exception(f"Error in liver data ingestion: {e}")