# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 14:56:27 2021

@author: Windows
"""

import numpy as np
from tensorflow import keras

# example of loading the vgg16 model
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
# load model
new_input = (32, 32, 3)
model = VGG16(include_top=False,input_shape=(32, 32, 3))

# add new classifier layers
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(100, activation='relu')(flat1)
output = Dense(10, activation='softmax')(class1)
# define new model
model = Model(inputs=model.inputs, outputs=output)
# summarize the model
model.summary()

from keras.utils.vis_utils import plot_model  
# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
import pydot_ng as pydot
pydot.find_graphviz()
plot_model(model, to_file='model_VGG_plot.png', show_shapes=True, show_layer_names=True)
# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# # x_train_100=x_train[0:10000,:,:,:]
# # y_train_100=y_train[0:10000,:]
# # x_test_100=x_test[0:10000,:,:,:]
# # y_test_100=y_test[0:10000,:]
# assert x_train.shape == (50000, 32, 32, 3)
# assert x_test.shape == (10000, 32, 32, 3)
# assert y_train.shape == (50000, 1)
# assert y_test.shape == (10000, 1)

# print("x_train shape:", x_train.shape)
# print(x_train.shape[0], "train samples")
# print(x_test.shape[0], "test samples")

# # Model / data parameters
# num_classes = 10
# input_shape = (32, 32, 3)

# # convert class vectors to binary class matrices
# y_train_100 = keras.utils.to_categorical(y_train, num_classes)
# y_test_100 = keras.utils.to_categorical(y_test, num_classes)

# batch_size = 128
# epochs = 50

# model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)

# ## 50k Data: Train Accu: 0.9159, Val accuracy: 0.7982
# ### 50k Data: Test Acc: 0.7944

# score = model.evaluate(x_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])