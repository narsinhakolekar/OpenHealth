import tensorflow as tf


class ModelEvaluation:

    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def evaluate(self, validation_generator):

        loss, accuracy = self.model.evaluate(
            validation_generator,
            verbose=1
        )

        print("\n==============================")
        print(f"Validation Loss     : {loss:.4f}")
        print(f"Validation Accuracy : {accuracy*100:.2f}%")
        print("==============================\n")

        return loss, accuracy