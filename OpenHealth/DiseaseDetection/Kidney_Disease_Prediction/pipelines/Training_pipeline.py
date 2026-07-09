# import os
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# # =========================
# # CONFIG
# # =========================
# IMAGE_SIZE = (150, 150)
# BATCH_SIZE = 32
# EPOCHS = 25

# # IMPORTANT:
# # This should point to the folder that contains the 4 class folders:
# # Cyst, Normal, Stone, Tumor
# DATASET_PATH = r"Notebook_Experiments\Data\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"

# MODEL_SAVE_PATH = r"Artifacts\Kidney_Disease\Kidney_Model.h5"

# os.makedirs(r"Artifacts\Kidney_Disease", exist_ok=True)

# # =========================
# # DATA GENERATORS
# # =========================
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2,
#     rotation_range=15,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     zoom_range=0.15,
#     shear_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# valid_datagen = ImageDataGenerator(
#     rescale=1./255,
#     validation_split=0.2
# )

# train_generator = train_datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='training',
#     shuffle=True
# )

# valid_generator = valid_datagen.flow_from_directory(
#     DATASET_PATH,
#     target_size=IMAGE_SIZE,
#     batch_size=BATCH_SIZE,
#     class_mode='categorical',
#     subset='validation',
#     shuffle=False
# )

# print("Class indices:", train_generator.class_indices)

# # =========================
# # MODEL
# # =========================
# model = Sequential([
#     tf.keras.Input(shape=(150, 150, 3)),

#     Conv2D(32, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(64, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(128, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Conv2D(256, (3,3), activation='relu'),
#     BatchNormalization(),
#     MaxPooling2D(2,2),

#     Flatten(),

#     Dense(256, activation='relu'),
#     Dropout(0.4),

#     Dense(128, activation='relu'),
#     Dropout(0.3),

#     Dense(4, activation='softmax')
# ])

# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )

# model.summary()

# # =========================
# # CALLBACKS
# # =========================
# early_stopping = EarlyStopping(
#     monitor='val_loss',
#     patience=5,
#     restore_best_weights=True
# )

# checkpoint = ModelCheckpoint(
#     MODEL_SAVE_PATH,
#     monitor='val_accuracy',
#     save_best_only=True,
#     mode='max',
#     verbose=1
# )

# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.3,
#     patience=2,
#     verbose=1
# )

# # =========================
# # TRAIN
# # =========================
# history = model.fit(
#     train_generator,
#     validation_data=valid_generator,
#     epochs=EPOCHS,
#     callbacks=[early_stopping, checkpoint, reduce_lr]
# )

# # =========================
# # SAVE FINAL MODEL
# # =========================
# model.save(MODEL_SAVE_PATH)
# print(f"\nKidney model saved at: {MODEL_SAVE_PATH}")

# # =========================
# # FINAL EVALUATION
# # =========================
# loss, acc = model.evaluate(valid_generator, verbose=1)
# print(f"Validation Accuracy: {acc:.4f}")
# print(f"Validation Loss: {loss:.4f}")

from OpenHealth.DiseaseDetection.Kidney_Disease_Prediction.components.Data_ingestion import DataIngestionConfig
from OpenHealth.DiseaseDetection.Kidney_Disease_Prediction.components.Model_trainer import ModelTrainer


if __name__ == "__main__":
    ingestion_config = DataIngestionConfig()
    trainer = ModelTrainer()
    trainer.initiate_model_training(ingestion_config.dataset_dir)