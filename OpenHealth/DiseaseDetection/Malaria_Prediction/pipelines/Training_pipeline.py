from OpenHealth.DiseaseDetection.Malaria_Prediction.components.Data_ingestion import (
    DataIngestionConfig
)

from OpenHealth.DiseaseDetection.Malaria_Prediction.components.Model_trainer import (
    ModelTrainer
)


def run_training():

    ingestion = DataIngestionConfig()

    trainer = ModelTrainer()

    trainer.initiate_model_training(
        ingestion.dataset_dir
    )


if __name__ == "__main__":
    run_training()