import os
import pickle
import numpy as np
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join(
        "Artifacts", "Liver_Disease", "Liver_Model.pkl"
    )


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X, y):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )

            model = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_split=4,
                min_samples_leaf=2,
                random_state=42
            )

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)

            print("\n==============================")
            print("LIVER MODEL TRAINING COMPLETE")
            print("==============================")
            print(f"Accuracy: {acc:.4f}")
            print("\nClassification Report:\n")
            print(classification_report(y_test, y_pred))

            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True
            )

            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(model, f)

            print(f"[INFO] Liver model saved at: {self.model_trainer_config.trained_model_file_path}")

            return acc

        except Exception as e:
            raise Exception(f"Error in liver model training: {e}")