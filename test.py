import numpy as np
import seaborn as sns
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.callbacks import ModelCheckpoint, EarlyStopping,TensorBoard
import tensorflow as tf
import os

data_gen = ImageDataGenerator()
target_size = (56,56)
batch_size = 128
class_mode='categorical'
train = data_gen.flow_from_directory("F:/Emotional-Detection-main/archive2/images/images/train",color_mode="grayscale",target_size = target_size, class_mode=class_mode,batch_size = batch_size)
valid = data_gen.flow_from_directory("F:/Emotional-Detection-main/archive2/images/images/validation",color_mode="grayscale",target_size = target_size, class_mode=class_mode,batch_size = batch_size)

# Initialising the CNN
model = Sequential()
model.add(tf.keras.Input(shape=(56, 56, 1)))
# 1 - Convolution
model.add(Conv2D(64,(3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd Convolution layer
model.add(Conv2D(128,(5,5), padding='same',activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd Convolution layer
model.add(Conv2D(512,(3,3), padding='same',activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th Convolution layer
model.add(Conv2D(512,(3,3), padding='same',activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Flattening
model.add(Flatten())

# Fully connected layer 1st layer
model.add(Dense(256,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

# Fully connected layer 2nd layer
model.add(Dense(512,activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(7, activation='softmax'))
# Setting up early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Learning rate scheduler configuration
lr_schedule = ExponentialDecay(initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)

# Optimizer setup with learning rate schedule
opt = Adam(learning_rate=lr_schedule)

# Model compilation
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Model checkpointing setup
checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# TensorBoard setup for performance visualization
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=1)

# Callbacks list
callbacks_list = [checkpoint,tensorboard]

# Model training
model.fit(train,
                        steps_per_epoch=train.samples/batch_size,
                        epochs=50,
                        validation_data = valid,
                        validation_steps = valid.samples//batch_size,
                        callbacks=callbacks_list
                        )
