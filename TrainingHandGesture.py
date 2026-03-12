"""
Hand Gesture Recognition using CNN
----------------------------------
This script trains a Convolutional Neural Network (CNN)
to recognize hand gestures from images.

Classes:
NONE, ONE, TWO, THREE, FOUR, FIVE
"""

# ===============================
# Import Required Libraries
# ===============================

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint


# ===============================
# Step 1: Build CNN Model
# ===============================

model = Sequential()

# First Convolution Layer
model.add(Conv2D(
    filters=32,
    kernel_size=(3, 3),
    input_shape=(256, 256, 1),
    activation='relu'
))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second Convolution Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third Convolution Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth Convolution Layer
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(units=150, activation='relu'))
model.add(Dropout(0.25))

# Output Layer (6 gesture classes)
model.add(Dense(units=6, activation='softmax'))


# ===============================
# Step 2: Compile Model
# ===============================

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# ===============================
# Step 3: Image Data Augmentation
# ===============================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=12,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)


# ===============================
# Step 4: Load Training Dataset
# ===============================

training_set = train_datagen.flow_from_directory(
    '../dataset/train',
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical'
)


# ===============================
# Step 5: Load Validation Dataset
# ===============================

val_set = val_datagen.flow_from_directory(
    '../dataset/test',
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical'
)


# ===============================
# Step 6: Training Callbacks
# ===============================

callbacks = [

    # Stop training if validation loss stops improving
    EarlyStopping(
        monitor='val_loss',
        patience=10
    ),

    # Save best model
    ModelCheckpoint(
        filepath='../models/model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
]


# ===============================
# Step 7: Train Model
# ===============================

model.fit(
    training_set,
    steps_per_epoch=37,
    epochs=5,
    validation_data=val_set,
    validation_steps=7,
    callbacks=callbacks
)


# ===============================
# Step 8: Save Model Architecture
# ===============================

model_json = model.to_json()

with open("../models/model.json", "w") as json_file:
    json_file.write(model_json)

print("Model training completed and saved successfully.")