import os
import pickle
import pandas as pd


class LiverData:
    def __init__(
        self,
        age,
        gender,
        total_bilirubin,
        direct_bilirubin,
        alkaline_phosphotase,
        alamine_aminotransferase,
        aspartate_aminotransferase,
        total_proteins,
        albumin,
        albumin_globulin_ratio
    ):
        self.age = age
        self.gender = gender
        self.total_bilirubin = total_bilirubin
        self.direct_bilirubin = direct_bilirubin
        self.alkaline_phosphotase = alkaline_phosphotase
        self.alamine_aminotransferase = alamine_aminotransferase
        self.aspartate_aminotransferase = aspartate_aminotransferase
        self.total_proteins = total_proteins
        self.albumin = albumin
        self.albumin_globulin_ratio = albumin_globulin_ratio

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Gender": [self.gender],
                "Total_Bilirubin": [self.total_bilirubin],
                "Direct_Bilirubin": [self.direct_bilirubin],
                "Alkaline_Phosphotase": [self.alkaline_phosphotase],
                "Alamine_Aminotransferase": [self.alamine_aminotransferase],
                "Aspartate_Aminotransferase": [self.aspartate_aminotransferase],
                "Total_Protiens": [self.total_proteins],
                "Albumin": [self.albumin],
                "Albumin_and_Globulin_Ratio": [self.albumin_globulin_ratio]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise Exception(f"Error in liver input dataframe creation: {e}")


class PredictLiver:
    def __init__(self):
        self.model_path = os.path.join("Artifacts", "Liver_Disease", "Liver_Model.pkl")
        self.preprocessor_path = os.path.join("Artifacts", "Liver_Disease", "Liver_Preprocessor.pkl")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Liver model not found at: {self.model_path}")

        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Liver preprocessor not found at: {self.preprocessor_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(self.preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)

    def predict(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            preds = self.model.predict(data_scaled)
            return preds

        except Exception as e:
            raise Exception(f"Error in liver prediction: {e}")

    def predict_proba(self, features):
        try:
            data_scaled = self.preprocessor.transform(features)
            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(data_scaled)
                return probs
            return None
        except Exception as e:
            raise Exception(f"Error in liver probability prediction: {e}")