import os
import sys
import pandas as pd

from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
from src.exception import customexception

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path = os.path.join(
        'Artifacts',
        'Heart_Disease',
        'Heart_Preprocessor.pkl'
    )


class DataTransformation:

    def __init__(self):

        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation(self):

        try:

            logging.info(
                "Heart Disease Prediction: Data Transformation initiated"
            )


            numerical_cols = [
                'Age',
                'RestingBP',
                'Cholesterol',
                'FastingBS',
                'MaxHR',
                'Oldpeak'
            ]


            categorical_cols = [
                'Sex',
                'ChestPainType',
                'RestingECG',
                'ExerciseAngina',
                'ST_Slope'
            ]


            numerical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )


            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            )


            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical_pipeline",
                        numerical_pipeline,
                        numerical_cols
                    ),
                    (
                        "categorical_pipeline",
                        categorical_pipeline,
                        categorical_cols
                    )
                ]
            )


            return preprocessor


        except Exception as e:

            logging.info(
                "Exception occurred in get_data_transformation"
            )

            raise customexception(e, sys)



    def initialize_data_transformation(
            self,
            train_path,
            test_path
    ):

        try:

            train_df = pd.read_csv(train_path)

            test_df = pd.read_csv(test_path)


            logging.info(
                "Heart Disease Prediction: Train and Test data loaded"
            )


            preprocessing_obj = self.get_data_transformation()


            target_column_name = "HeartDisease"


            X_train = train_df.drop(
                columns=[target_column_name],
                axis=1
            )

            y_train = train_df[target_column_name]


            X_test = test_df.drop(
                columns=[target_column_name],
                axis=1
            )

            y_test = test_df[target_column_name]


            logging.info(
                "Heart Disease Prediction: Input and target separated"
            )


            X_train = preprocessing_obj.fit_transform(
                X_train
            )


            X_test = preprocessing_obj.transform(
                X_test
            )


            logging.info(
                "Heart Disease Prediction: Preprocessing completed"
            )


            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            logging.info(
                "Heart Preprocessor saved successfully"
            )


            return (
                X_train,
                X_test,
                y_train,
                y_test
            )


        except Exception as e:

            logging.info(
                "Exception occurred in initialize_data_transformation"
            )

            raise customexception(e, sys)