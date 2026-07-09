import os
import pickle
from sklearn.metrics import accuracy_score


def save_object(file_path, obj):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)


def load_object(file_path):
    with open(file_path, "rb") as file_obj:
        return pickle.load(file_obj)


def evaluate_model(X_train, y_train, X_test, y_test, models):
    model_report = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_test_pred = model.predict(X_test)
        test_model_score = accuracy_score(y_test, y_test_pred)

        model_report[model_name] = test_model_score

    return model_report