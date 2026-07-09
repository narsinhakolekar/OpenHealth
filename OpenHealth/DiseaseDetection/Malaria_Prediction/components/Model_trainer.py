import os
import json
import numpy as np
import tensorflow as tf

from dataclasses import dataclass

from tensorflow.keras import layers
from tensorflow.keras import models

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


@dataclass
class ModelTrainerConfig:

    image_size = (224,224)

    batch_size = 32

    epochs_head = 12

    epochs_finetune = 15

    artifact_dir = os.path.join(
        "Artifacts",
        "Malaria"
    )

    model_path = os.path.join(
        artifact_dir,
        "Malaria_Model.keras"
    )

    class_file = os.path.join(
        artifact_dir,
        "class_names.npy"
    )

    history_file = os.path.join(
        artifact_dir,
        "training_history.json"
    )


class ModelTrainer:

    def __init__(self):

        self.config = ModelTrainerConfig()


    def initiate_model_training(self,dataset_dir):

        os.makedirs(self.config.artifact_dir,exist_ok=True)

        train_datagen = ImageDataGenerator(

            preprocessing_function=preprocess_input,

            validation_split=0.20,

            rotation_range=30,

            zoom_range=0.20,

            shear_range=0.15,

            width_shift_range=0.20,

            height_shift_range=0.20,

            horizontal_flip=True,

            brightness_range=[0.9,1.1]

        )

        validation_datagen = ImageDataGenerator(

            preprocessing_function=preprocess_input,

            validation_split=0.20

        )

        train_generator = train_datagen.flow_from_directory(

            dataset_dir,

            target_size=self.config.image_size,

            batch_size=self.config.batch_size,

            subset="training",

            shuffle=True,

            class_mode="categorical"

        )

        validation_generator = validation_datagen.flow_from_directory(

            dataset_dir,

            target_size=self.config.image_size,

            batch_size=self.config.batch_size,

            subset="validation",

            shuffle=False,

            class_mode="categorical"

        )

        print(train_generator.class_indices)

        class_names = list(train_generator.class_indices.keys())

        np.save(self.config.class_file,class_names)

        base_model = MobileNetV2(

            weights="imagenet",

            include_top=False,

            input_shape=(224,224,3)

        )

        base_model.trainable=False

        inputs=tf.keras.Input(shape=(224,224,3))

        x=base_model(inputs,training=False)

        x=layers.GlobalAveragePooling2D()(x)

        x=layers.Dropout(0.30)(x)

        x=layers.Dense(128,activation="relu")(x)

        x=layers.Dropout(0.20)(x)

        outputs=layers.Dense(2,activation="softmax")(x)

        model=models.Model(inputs,outputs)

        model.compile(

            optimizer=tf.keras.optimizers.Adam(1e-3),

            loss="categorical_crossentropy",

            metrics=["accuracy"]

        )

        callbacks=[

            EarlyStopping(

                monitor="val_accuracy",

                patience=5,

                restore_best_weights=True

            ),

            ReduceLROnPlateau(

                monitor="val_loss",

                factor=0.30,

                patience=2,

                verbose=1

            ),

            ModelCheckpoint(

                self.config.model_path,

                monitor="val_accuracy",

                save_best_only=True,

                verbose=1

            )

        ]

        print("\nStage-1 Training\n")

        history1=model.fit(

            train_generator,

            validation_data=validation_generator,

            epochs=self.config.epochs_head,

            callbacks=callbacks

        )

        print("\nFine Tuning\n")

        base_model.trainable=True

        for layer in base_model.layers[:-30]:

            layer.trainable=False

        model.compile(

            optimizer=tf.keras.optimizers.Adam(1e-5),

            loss="categorical_crossentropy",

            metrics=["accuracy"]

        )

        history2=model.fit(

            train_generator,

            validation_data=validation_generator,

            epochs=self.config.epochs_finetune,

            callbacks=callbacks

        )

        best_model=tf.keras.models.load_model(

            self.config.model_path

        )

        loss,acc=best_model.evaluate(

            validation_generator,

            verbose=1

        )

        history={

            "stage1":history1.history,

            "stage2":history2.history,

            "final_accuracy":float(acc),

            "final_loss":float(loss)

        }

        with open(self.config.history_file,"w") as f:

            json.dump(history,f,indent=4)

        print("\n===================================")
        print("Model Saved Successfully")
        print(self.config.model_path)
        print(f"Validation Accuracy : {acc*100:.2f}%")
        print("===================================\n")

        return best_model