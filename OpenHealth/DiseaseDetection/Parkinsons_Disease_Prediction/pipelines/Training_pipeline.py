from OpenHealth.DiseaseDetection.Parkinsons_Disease_Prediction.components.Data_ingestion import DataIngestion
from OpenHealth.DiseaseDetection.Parkinsons_Disease_Prediction.components.Data_transformation import DataTransformation
from OpenHealth.DiseaseDetection.Parkinsons_Disease_Prediction.components.Model_trainer import ModelTrainer

# Data Ingestion
obj = DataIngestion()
train_data_path, test_data_path = obj.initiate_data_ingestion()

# Data Transformation
data_transformation = DataTransformation()
x_train, y_train, x_test, y_test = data_transformation.initialize_data_transformation(train_data_path, test_data_path)

# Model Training
model_trainer_obj = ModelTrainer()
model_trainer_obj.initiate_model_training(x_train, x_test, y_train, y_test)