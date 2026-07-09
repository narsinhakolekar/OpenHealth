import os
import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join(
        "Artifacts", "Liver_Disease", "Liver_Preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_features = [
                "Age",
                "Total_Bilirubin",
                "Direct_Bilirubin",
                "Alkaline_Phosphotase",
                "Alamine_Aminotransferase",
                "Aspartate_Aminotransferase",
                "Total_Protiens",
                "Albumin",
                "Albumin_and_Globulin_Ratio"
            ]

            categorical_features = ["Gender"]

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", numeric_pipeline, numeric_features),
                    ("cat_pipeline", categorical_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise Exception(f"Error in liver transformer object: {e}")

    def initiate_data_transformation(self, raw_data_path):
        try:
            df = pd.read_csv(raw_data_path)
            df.columns = [col.strip() for col in df.columns]

            # Replace ? with NaN if present
            df.replace("?", np.nan, inplace=True)

            # Convert known numeric columns
            numeric_cols = [
                "Age",
                "Total_Bilirubin",
                "Direct_Bilirubin",
                "Alkaline_Phosphotase",
                "Alamine_Aminotransferase",
                "Aspartate_Aminotransferase",
                "Total_Protiens",
                "Albumin",
                "Albumin_and_Globulin_Ratio",
                "Dataset"
            ]

            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            # ILPD target mapping:
            # 1 => liver disease
            # 2 => no liver disease
            # convert to 1/0
            df["Dataset"] = df["Dataset"].replace({1: 1, 2: 0})

            df = df.dropna(subset=["Dataset"])

            X = df.drop(columns=["Dataset"])
            y = df["Dataset"].astype(int)

            preprocessor = self.get_data_transformer_object()

            X_transformed = preprocessor.fit_transform(X)

            os.makedirs(
                os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),
                exist_ok=True
            )

            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            print("Liver data transformation completed.")
            print("Saved preprocessor to:", self.data_transformation_config.preprocessor_obj_file_path)

            return X_transformed, y, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise Exception(f"Error in liver data transformation: {e}")