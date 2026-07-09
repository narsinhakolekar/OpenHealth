from OpenHealth.DiseaseDetection.Liver_Disease_Prediction.components.Data_ingestion import DataIngestion
from OpenHealth.DiseaseDetection.Liver_Disease_Prediction.components.Data_transformation import DataTransformation
from OpenHealth.DiseaseDetection.Liver_Disease_Prediction.components.Model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        raw_data_path = DataIngestion().initiate_data_ingestion()
        X, y, _ = DataTransformation().initiate_data_transformation(raw_data_path)
        score = ModelTrainer().initiate_model_trainer(X, y)
        return score


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()