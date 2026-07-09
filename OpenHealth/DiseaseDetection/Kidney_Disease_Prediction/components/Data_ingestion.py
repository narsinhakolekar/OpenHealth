import os
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    dataset_dir: str = os.path.join(
        "Notebook_Experiments",
        "Data",
        "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
        "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    )