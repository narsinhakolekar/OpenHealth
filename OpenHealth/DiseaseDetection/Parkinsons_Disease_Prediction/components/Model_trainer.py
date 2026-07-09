import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

from src.logger import logging
from src.exception import customexception
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("Artifacts", "Parkinsons_Disease", "Parkinsons_Model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, x_train, x_test, y_train, y_test):
        try:
            models = {
                "LR": LogisticRegression(max_iter=1000),
                "NB": GaussianNB(),
                "XGB": XGBClassifier(
                    learning_rate=0.01,
                    n_estimators=25,
                    max_depth=15,
                    gamma=0.6,
                    subsample=0.52,
                    colsample_bytree=0.6,
                    seed=27,
                    reg_lambda=2,
                    booster='dart',
                    colsample_bylevel=0.6,
                    colsample_bynode=0.5
                ),
                "RF": RandomForestClassifier(),
                "GB": GradientBoostingClassifier(),
                "DT": DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=6),
                "KNN": KNeighborsClassifier(),
                "EXT": ExtraTreesClassifier()
            }

            model_report = evaluate_model(x_train, y_train, x_test, y_test, models)

            print(model_report)
            print("\n====================================================================================\n")
            logging.info(f"Parkinson's Disease: Model Report -> {model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"Parkinson's Disease: Best Model Found -> {best_model_name}, Accuracy Score: {best_model_score}")
            print("\n====================================================================================\n")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Exception occurred in Model Training")
            raise customexception(e, sys)