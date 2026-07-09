import os
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    dataset_dir = os.path.join(
        "Notebook_Experiments",
        "Data",
        "cell_images"
    )