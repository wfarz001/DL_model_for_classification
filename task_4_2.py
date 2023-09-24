# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 19:27:04 2021

@author: Windows
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import pathlib
import glob
import os
from PIL import Image
#import SimpleITK as sitk
import cv2
from tensorflow import keras
from tensorflow.keras import layers

cancer_tr=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Cancer/Resized_Images/Training')

cancer_train=[]

for file in cancer_tr.iterdir():
        image=cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
        cancer_train.append(image) 
        
y_tr_cancer= np.array([1]*42)   
     
cancer_train=np.array(cancer_train)     



non_cancer_tr=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Non_Cancer/Resized_Images/Testing') 

non_cancer_train=[]

  
for file in non_cancer_tr.iterdir():
        image=cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
        non_cancer_train.append(image)     
        
y_tr_non_cancer=np.array([0]*162)

non_cancer_train=np.array(non_cancer_train)


X_train=np.concatenate((cancer_train,non_cancer_train),axis=0)
y_train=np.concatenate((y_tr_cancer,y_tr_non_cancer),axis=0)



from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers

from keras.layers import BatchNormalization


from keras.utils.np_utils import to_categorical

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 2)


model=Sequential()

#model.add(Lambda(standardize,input_shape=(28,28,1)))    
model.add(Conv2D(filters=8, kernel_size = (3,3), activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=16, kernel_size = (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size = (3,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())

    
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(100,activation="relu"))

model.add(Dense(2,activation="sigmoid"))
    
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 1
epochs = 15

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

model.save("Task_4_model.h5")