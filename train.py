#!/usr/bin/env python

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Input, Activation, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
from keras.optimizers import SGD, Adagrad

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import h5py

import glob
import os

NUMBER_OF_CLASSES  = 13
DATA_FOLDER = "/mount/SDC/casava/split-data/output"
img_width, img_height = 224, 224
crop_width, crop_height = 224, 224

train_data_dir = DATA_FOLDER + '/train'
validation_data_dir = DATA_FOLDER + '/test'
nb_train_samples = 9284
nb_validation_samples = 2389
nb_epoch = 50

base_model = VGG16(weights='imagenet', input_tensor=Input((3, 224, 224)),include_top=False)
# add a global spatial average pooling layer
x = base_model.output
# x = GlobalAveragePooling2D()(x)
x = Flatten(name="flatten")(x)
x = Dense(4096, activation='relu', name="dense_1")(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name="dense_2")(x)
x = Dropout(0.5)(x)
x = Dense(NUMBER_OF_CLASSES, name="dense_3")(x)
predictions = Activation("softmax", name="activation")(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)
_opt = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=_opt, loss='categorical_crossentropy', metrics=["accuracy"])



# Add checkpoints
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='categorical')


model.fit_generator(
        train_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        callbacks=callbacks_list)
