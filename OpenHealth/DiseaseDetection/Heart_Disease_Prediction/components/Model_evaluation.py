import os
import mlflow
import mlflow.sklearn

from urllib.parse import urlparse

from src.utils import load_object

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ModelEvaluation:

    def __init__(self):
        pass


    def eval_metrics(self, actual, pred):

        accuracy = accuracy_score(actual, pred)

        precision = precision_score(
            actual,
            pred,
            zero_division=0
        )

        recall = recall_score(
            actual,
            pred,
            zero_division=0
        )

        f1 = f1_score(
            actual,
            pred,
            zero_division=0
        )

        return accuracy, precision, recall, f1



    def initiate_model_evaluation(
            self,
            x_train,
            x_test,
            y_train,
            y_test
    ):

        try:

            model_path = os.path.join(
                "Artifacts",
                "Heart_Disease",
                "Heart_Model.pkl"
            )


            model = load_object(model_path)


            # MLflow tracking
            mlflow.set_tracking_uri(
                "sqlite:///mlflow.db"
            )


            with mlflow.start_run():


                predictions = model.predict(
                    x_test
                )


                accuracy, precision, recall, f1 = self.eval_metrics(
                    y_test,
                    predictions
                )


                mlflow.log_metric(
                    "Testing Accuracy",
                    accuracy
                )

                mlflow.log_metric(
                    "Precision Score",
                    precision
                )

                mlflow.log_metric(
                    "Recall Score",
                    recall
                )

                mlflow.log_metric(
                    "F1 Score",
                    f1
                )


                mlflow.sklearn.log_model(
                    model,
                    "Model"
                )


                print("Model evaluation completed")
                print(
                    f"Accuracy: {accuracy}"
                )



        except Exception as e:

            raise e