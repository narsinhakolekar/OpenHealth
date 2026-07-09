import sys
import pandas as pd
from src.logger import logging
from src.exception import customexception


class DataTransformation:
    def __init__(self):
        pass

    def initialize_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Parkinson's Disease: Train and test data loaded")

            target_column_name = "status"

            selected_columns = [
                "MDVP:Fo(Hz)",
                "MDVP:Fhi(Hz)",
                "MDVP:Flo(Hz)",
                "MDVP:Jitter(%)",
                "RPDE",
                "DFA",
                "spread2",
                "D2"
            ]

            x_train = train_df[selected_columns]
            y_train = train_df[target_column_name]

            x_test = test_df[selected_columns]
            y_test = test_df[target_column_name]

            logging.info("Parkinson's Disease: Data transformation completed")

            return x_train, y_train, x_test, y_test

        except Exception as e:
            logging.info("Exception occurred in Data Transformation")
            raise customexception(e, sys)