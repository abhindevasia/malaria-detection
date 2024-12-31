# Importing Libraries

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
from PIL import Image, ImageOps

# Preparing datasets and preprocessing for model creation
# Paths to dataset directories
inf_train_inf_dir = 'data/train/Parasitized/'
inf_train_norm_dir = 'data/train/Uninfected/'

# Load and preprocess infected images
infected_images = []
for img in os.listdir(inf_train_inf_dir):
    if img.endswith(".png"):
        image = cv2.imread(os.path.join(inf_train_inf_dir, img))
        resized_image = cv2.resize(image, (64, 64))
        infected_images.append(resized_image)

train_infected = np.array(infected_images, dtype='float32') / 255.0

# Load and preprocess uninfected images
uninfected_images = []
for img in os.listdir(inf_train_norm_dir):
    if img.endswith(".png"):
        image = cv2.imread(os.path.join(inf_train_norm_dir, img))
        resized_image = cv2.resize(image, (64, 64))
        uninfected_images.append(resized_image)

train_uninfected = np.array(uninfected_images, dtype='float32') / 255.0

# Create labels for uninfected (0) and infected (1)
labels_uninfected = np.zeros((train_uninfected.shape[0], 1))  # Label 0 for uninfected
labels_infected = np.ones((train_infected.shape[0], 1))       # Label 1 for infected

# Split data into training and testing sets
Xtrain = np.concatenate((train_uninfected[8717:], train_infected[8717:]))
Xtest = np.concatenate((train_uninfected[:8717], train_infected[:8717]))
ytrain = np.concatenate((labels_uninfected[8717:], labels_infected[8717:]))
ytest = np.concatenate((labels_uninfected[:8717], labels_infected[:8717]))

# Flatten labels for use in ImageDataGenerator
ytrain = ytrain.flatten()
ytest = ytest.flatten()

# Data Augmentation: Define the data augmentation techniques
datagen = ImageDataGenerator(
    rotation_range=20,       # Random rotations
    width_shift_range=0.2,   # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    shear_range=0.2,         # Random shear
    zoom_range=0.2,          # Random zoom
    horizontal_flip=True,    # Random horizontal flip
    fill_mode='nearest'      # Fill missing pixels after transformation
)

# Augment the training data
train_data_gen = datagen.flow(Xtrain, ytrain, batch_size=32)

# Building Model
# Model Definition: Define the CNN architecture with improvements
model = Sequential()

# Convolutional layers with Batch Normalization and MaxPooling
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output of the convolutional layers
model.add(Flatten())

# Fully connected layers with Dropout to reduce overfitting
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout layer to prevent overfitting

# Output layer for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks: EarlyStopping and ReduceLROnPlateau to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# Train the model
history = model.fit(
    train_data_gen,
    epochs=50,
    validation_data=(Xtest, ytest),
    batch_size=32,
    callbacks=[early_stopping, reduce_lr]
)

# Plot the training and validation accuracy/loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

#saving Model
model.save('models/malaria_detect_model.keras')
