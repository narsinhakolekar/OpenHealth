import os
import tensorflow as tf
from dataclasses import dataclass
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


@dataclass
class ModelTrainerConfig:
    image_size = (224, 224)
    batch_size = 32
    epochs_head = 10          # first phase: train classifier head
    epochs_finetune = 10      # second phase: fine tune top layers
    model_path = os.path.join("Artifacts", "Kidney_Disease", "Kidney_Model.h5")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_training(self, dataset_dir):
        os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)

        # -----------------------------
        # Data generators
        # -----------------------------
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2,
            rotation_range=20,
            width_shift_range=0.15,
            height_shift_range=0.15,
            zoom_range=0.15,
            shear_range=0.1,
            horizontal_flip=True
        )

        valid_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            validation_split=0.2
        )

        train_generator = train_datagen.flow_from_directory(
            dataset_dir,
            target_size=self.config.image_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )

        valid_generator = valid_datagen.flow_from_directory(
            dataset_dir,
            target_size=self.config.image_size,
            batch_size=self.config.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )

        print("\nClass indices:", train_generator.class_indices)

        # -----------------------------
        # Base MobileNetV2 model
        # -----------------------------
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False

        # -----------------------------
        # Classification head
        # -----------------------------
        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(4, activation='softmax')(x)

        model = models.Model(inputs, outputs)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\n================ STAGE 1: TRAIN HEAD ================\n")
        model.summary()

        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=2,
                verbose=1
            ),
            ModelCheckpoint(
                self.config.model_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        history1 = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=self.config.epochs_head,
            callbacks=callbacks
        )

        # -----------------------------
        # Fine-tuning
        # -----------------------------
        print("\n================ STAGE 2: FINE-TUNING ================\n")

        base_model.trainable = True

        # Freeze lower layers, train only top layers
        for layer in base_model.layers[:-40]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        history2 = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=self.config.epochs_finetune,
            callbacks=callbacks
        )

        # Load best saved model
        best_model = tf.keras.models.load_model(self.config.model_path)

        # Final evaluation
        loss, acc = best_model.evaluate(valid_generator, verbose=1)
        print("\n====================================================")
        print(f"Final Validation Accuracy: {acc * 100:.2f}%")
        print(f"Best model saved at: {self.config.model_path}")
        print("====================================================\n")

        return best_model, history1, history2