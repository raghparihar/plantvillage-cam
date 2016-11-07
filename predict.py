#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function

import keras_image
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Input, Activation, GlobalAveragePooling2D
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras import backend as K
from keras.optimizers import SGD
#
from keras.callbacks import ModelCheckpoint
# from keras.preprocessing.image import ImageDataGenerator
#
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import h5py
from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D

import glob
import os
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import json

import uuid


OUTPUT_DIRECTORY = "/mount/SDC/PROJECTION_RESULTS"
NUMBER_OF_CLASSES  = 13
DATA_FOLDER = "/mount/SDC/casava/split-data/output"
img_width, img_height = 224, 224
crop_width, crop_height = 224, 224

train_data_dir = DATA_FOLDER + '/train'
validation_data_dir = DATA_FOLDER + '/test'
nb_train_samples = 9284
nb_validation_samples = 2389
nb_epoch = 50

#Build actual Prediction model
base_model = VGG16(weights=None, input_tensor=Input((3, 224, 224)),include_top=False)
x = base_model.output
x = Flatten(name="flatten")(x)
x = Dense(4096, activation='relu', name="dense_1")(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu', name="dense_2")(x)
x = Dropout(0.5)(x)
x = Dense(NUMBER_OF_CLASSES, name="dense_3")(x)
predictions = Activation("softmax", name="activation")(x)
# this is the model we will train
model = Model(input=base_model.input, output=predictions)
WEIGHTS_PATH = "./weights-improvement-39-0.85.hdf5"
model.load_weights(WEIGHTS_PATH)

#Build force-convolutionalized model for Projection
base_model = VGG16(weights=None, input_tensor=Input((3, None, None)),include_top=False)
x = base_model.output
x = Convolution2D(4096,7,7,activation="relu",name="dense_1")(x)
x = Convolution2D(4096,1,1,activation="relu",name="dense_2")(x)
x = Convolution2D(NUMBER_OF_CLASSES,1,1,name="dense_3")(x)
_output = Softmax4D(axis=1,name="softmax")(x)
_projection_model = Model(input=base_model.input, output=_output)

#Copy weights from the prediction model to the projection model
for layer in _projection_model.layers:
    if layer.name.split("_")[-1][:-1] == "conv":
        orig_layer = model.get_layer(layer.name)
        layer.set_weights(orig_layer.get_weights())
        print("Copying layer weights for :", layer.name)
    elif layer.name.startswith("dense_"):
        orig_layer = model.get_layer(layer.name)
        W,b = orig_layer.get_weights()
        n_filter,previous_filter,ax1,ax2 = layer.get_weights()[0].shape
        new_W = W.reshape((previous_filter,ax1,ax2,n_filter))
        new_W = new_W.transpose((3,0,1,2))
        new_W = new_W[:,:,::-1,::-1]
        layer.set_weights([new_W,b])
        print("Adapting and copying layer weights for :", layer.name)


#Compile both the models
model.compile(optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=["accuracy"])
_projection_model.compile(optimizer=SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=["accuracy"])


ORIGINAL_IMAGES_PATH = "/mount/SDC/casava/new_data/Dropbox/casava"
DATA_FOLDER = "/mount/SDC/casava/split-data/output/test-trimmed"

"""
Build Original images map
"""
_high_res_map = {}
for _file in glob.glob(ORIGINAL_IMAGES_PATH+"/*/*.JPG"):
    _fileName = _file.split("/")[-1]
    if _fileName in _high_res_map.keys():
        print("CONFLICT : ",_file)
    else:
        _high_res_map[_fileName] = _file

classMap = os.listdir(DATA_FOLDER)
TEST_FILES = glob.glob(DATA_FOLDER+"/*/*.JPG")
X = []
X_PATH = []
Y = []
Y_READABLE = []
X_RESIZED = []

for _image in TEST_FILES:
    _class = _image.split("/")[-2]
    _fileName = _image.split("/")[-1]
    if _fileName not in _high_res_map.keys():
        print("HIGHRESMAP CONFLICT ::: ", _image)
        os.remove(_image)
    else:
        ORIGINAL_FILE = _high_res_map[_fileName]
        #Load image
        X_PATH.append(ORIGINAL_FILE)
        X_RESIZED.append(_image)
        Y_READABLE.append(_class)

Y = [classMap.index(y) for y in Y_READABLE]

BATCH_SIZE = 32
test_datagen = keras_image.ImageDataGenerator(rescale=1./255)
# test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        high_res_map=_high_res_map,
        high_res_target_size=(2000, 2000))


PREDS = []
TRUE_LABELS = []

for iteration_number, data in enumerate(validation_generator):
    y_pred = model.predict(data[0])
    y_pred = np.array([x.argmax() for x in y_pred]).tolist()
    y_true = data[1]
    y_true = np.array([x.argmax() for x in y_true]).tolist()
    high_res_images = data[2]
    batch_filenames = data[3]

    PREDS += y_pred
    TRUE_LABELS += y_true

    print("Y-PRED : ",y_pred)
    print("Y-TRUE : ", y_true)

    for _idx, _high_res_img in enumerate(high_res_images):
        print("Predicted : ", classMap[y_pred[_idx]], " Actual :", classMap[y_true[_idx]])
        _projection = _projection_model.predict(_high_res_img.reshape((1,) + _high_res_img.shape))
        basewidth = 400
        hsize = 400
        if classMap[y_pred[_idx]] == classMap[y_true[_idx]]:
            result = "CORRECT"
            proj = _projection[0, y_pred[_idx]]
            resized_proj = np.asarray(Image.fromarray(proj).resize((basewidth, hsize), Image.ANTIALIAS))
            DIR = OUTPUT_DIRECTORY+"/"+result+"/"+classMap[y_true[_idx]]+"/"+str(uuid.uuid4())+".JPG"
            try:
                os.makedirs(DIR)
            except:
                pass
            PATH_TO_PROJECTION = DIR + "/PREDICTED_PROJECTION.png"
            plt.imsave(PATH_TO_PROJECTION, resized_proj)
            PATH_TO_ORIGINAL_IMAGE = DIR+"/ORIGINAL_IMAGE.png"
            keras_image.array_to_img(_high_res_img).resize((basewidth, hsize), Image.ANTIALIAS).save(PATH_TO_ORIGINAL_IMAGE)
            _result_object = {"predicted":classMap[y_pred[_idx]], "actual": classMap[y_true[_idx]], "filepath": batch_filenames[_idx]}
            with open(DIR+'/result.json', 'w') as outfile:
                json.dump(_result_object, outfile)
        else:
            result = "INCORRECT"
            proj = _projection[0, y_pred[_idx]]
            resized_proj = np.asarray(Image.fromarray(proj).resize((basewidth, hsize), Image.ANTIALIAS))
            DIR = OUTPUT_DIRECTORY+"/"+result+"/"+classMap[y_true[_idx]]+"/"+str(uuid.uuid4())+".JPG"
            try:
                os.makedirs(DIR)
            except:
                pass
            PATH_TO_PROJECTION = DIR+"/PREDICTED_PROJECTION.png"
            plt.imsave(PATH_TO_PROJECTION, resized_proj)
            #Render projection for the actual class of the Image
            proj = _projection[0, y_true[_idx]]
            resized_proj = np.asarray(Image.fromarray(proj).resize((basewidth, hsize), Image.ANTIALIAS))
            PATH_TO_PROJECTION = DIR+"/TRUELABELs_PROJECTION.png"
            plt.imsave(PATH_TO_PROJECTION, resized_proj)

            PATH_TO_ORIGINAL_IMAGE = DIR+"/ORIGINAL_IMAGE.png"
            keras_image.array_to_img(_high_res_img).resize((basewidth, hsize), Image.ANTIALIAS).save(PATH_TO_ORIGINAL_IMAGE)
            _result_object = {"predicted":classMap[y_pred[_idx]], "actual": classMap[y_true[_idx]], "filepath": batch_filenames[_idx]}
            with open(DIR+'/result.json', 'w') as outfile:
                json.dump(_result_object, outfile)

    if (iteration_number+1)*BATCH_SIZE >= len(Y):
        break

from sklearn.metrics import classification_report
print("Generating Classification Report")
print(classification_report(TRUE_LABELS, PREDS))
