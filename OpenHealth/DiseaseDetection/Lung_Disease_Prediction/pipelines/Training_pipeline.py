import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Optional logging compatibility with your project
try:
    from src.logger import logging
except:
    import logging
    logging.basicConfig(level=logging.INFO)

# =========================
# Configuration
# =========================
BASE_DIR = os.getcwd()

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "Notebook_Experiments",
    "Data",
    "Lung Disease Dataset",
    "train"
)

VAL_DIR = os.path.join(
    BASE_DIR,
    "Notebook_Experiments",
    "Data",
    "Lung Disease Dataset",
    "val"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "Notebook_Experiments",
    "Data",
    "Lung Disease Dataset",
    "test"
)

ARTIFACT_DIR = os.path.join(BASE_DIR, "Artifacts", "Lung_Disease")
MODEL_PATH = os.path.join(ARTIFACT_DIR, "Lung_Model.h5")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 5
SEED = 42


def verify_dataset():
    required_dirs = [TRAIN_DIR, VAL_DIR, TEST_DIR]
    for d in required_dirs:
        if not os.path.exists(d):
            raise FileNotFoundError(f"Dataset folder not found: {d}")

    print("\n================ Lung Dataset Check ================")
    print("Train:", TRAIN_DIR)
    print("Val  :", VAL_DIR)
    print("Test :", TEST_DIR)
    print("====================================================\n")


def get_generators():
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        shear_range=0.10,
        horizontal_flip=True
    )

    # Validation / Test only rescaling
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    val_generator = test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator, test_generator


def build_model(num_classes=5):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Freeze base initially
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.35),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base_model


def fine_tune_model(model, base_model):
    # Unfreeze top layers of MobileNetV2 for fine-tuning
    base_model.trainable = True

    # Freeze earlier layers, unfreeze later layers only
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train():
    try:
        verify_dataset()
        os.makedirs(ARTIFACT_DIR, exist_ok=True)

        train_generator, val_generator, test_generator = get_generators()

        print("Class indices:", train_generator.class_indices)

        model, base_model = build_model(num_classes=train_generator.num_classes)

        callbacks = [
            ModelCheckpoint(
                MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]

        print("\n================ Stage 1: Initial Training ================\n")
        history1 = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=8,
            callbacks=callbacks
        )

        print("\n================ Stage 2: Fine Tuning ================\n")
        model = fine_tune_model(model, base_model)

        history2 = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=EPOCHS,
            initial_epoch=history1.epoch[-1] + 1,
            callbacks=callbacks
        )

        # Load best saved model
        best_model = tf.keras.models.load_model(MODEL_PATH)

        print("\n================ Final Evaluation ================\n")
        test_loss, test_acc = best_model.evaluate(test_generator, verbose=1)

        print("====================================================")
        print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
        print(f"Best model saved at: {MODEL_PATH}")
        print("====================================================\n")

        logging.info(f"Lung model saved at: {MODEL_PATH}")
        logging.info(f"Final Lung Test Accuracy: {test_acc * 100:.2f}%")

    except Exception as e:
        print("Error in Lung training pipeline:", e)
        raise e


if __name__ == "__main__":
    train()