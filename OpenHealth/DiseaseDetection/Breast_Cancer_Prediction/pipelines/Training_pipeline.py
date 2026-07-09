from OpenHealth.DiseaseDetection.Breast_Cancer_Prediction.components.Data_ingestion import DataIngestion
from OpenHealth.DiseaseDetection.Breast_Cancer_Prediction.components.Data_transformation import DataTransformation
from OpenHealth.DiseaseDetection.Breast_Cancer_Prediction.components.Model_trainer import ModelTrainer

# Data ingestion
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# Data transformation
data_transformation = DataTransformation()
train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

# Model training
model_trainer_obj = ModelTrainer()
model_trainer_obj.initate_model_training(train_arr, test_arr)