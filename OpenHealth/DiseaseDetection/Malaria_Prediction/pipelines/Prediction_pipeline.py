import os
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class PredictionPipeline:

    def __init__(self):

        self.model_path = os.path.join(
            "Artifacts",
            "Malaria",
            "Malaria_Model.keras"
        )

        self.class_path = os.path.join(
            "Artifacts",
            "Malaria",
            "class_names.npy"
        )

        self.model = load_model(self.model_path)

        self.class_names = np.load(
            self.class_path,
            allow_pickle=True
        )

    def predict(self, image_path):

        img = image.load_img(
            image_path,
            target_size=(224,224)
        )

        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        img = preprocess_input(img)

        prediction = self.model.predict(img, verbose=0)

        probabilities = prediction[0]

        predicted_index = np.argmax(probabilities)

        predicted_class = self.class_names[predicted_index]

        confidence = float(probabilities[predicted_index]) * 100

        probability_table = {}

        for i, cls in enumerate(self.class_names):

            probability_table[cls] = round(
                float(probabilities[i]) * 100,
                2
            )

        return {

            "prediction": predicted_class,

            "confidence": round(confidence,2),

            "probabilities": probability_table

        }