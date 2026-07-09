import os
import sys
import pandas as pd

from src.utils import load_object
from src.exception import customexception



class PredictPipeline:

    def predict(self,features):

        try:

            preprocessor_path=os.path.join(
                "Artifacts",
                "Heart_Disease",
                "Heart_Preprocessor.pkl"
            )

            model_path=os.path.join(
                "Artifacts",
                "Heart_Disease",
                "Heart_Model.pkl"
            )


            preprocessor=load_object(preprocessor_path)

            model=load_object(model_path)


            data=preprocessor.transform(features)


            prediction=model.predict(data)


            return prediction


        except Exception as e:
            raise customexception(e,sys)





class CustomData:

    def __init__(
        self,
        Age,
        Sex,
        ChestPainType,
        RestingBP,
        Cholesterol,
        FastingBS,
        RestingECG,
        MaxHR,
        ExerciseAngina,
        Oldpeak,
        ST_Slope
    ):

        self.Age=Age
        self.Sex=Sex
        self.ChestPainType=ChestPainType
        self.RestingBP=RestingBP
        self.Cholesterol=Cholesterol
        self.FastingBS=FastingBS
        self.RestingECG=RestingECG
        self.MaxHR=MaxHR
        self.ExerciseAngina=ExerciseAngina
        self.Oldpeak=Oldpeak
        self.ST_Slope=ST_Slope




    def get_data_as_dataframe(self):

        data={

            "Age":[self.Age],
            "Sex":[self.Sex],
            "ChestPainType":[self.ChestPainType],
            "RestingBP":[self.RestingBP],
            "Cholesterol":[self.Cholesterol],
            "FastingBS":[self.FastingBS],
            "RestingECG":[self.RestingECG],
            "MaxHR":[self.MaxHR],
            "ExerciseAngina":[self.ExerciseAngina],
            "Oldpeak":[self.Oldpeak],
            "ST_Slope":[self.ST_Slope]

        }


        return pd.DataFrame(data)