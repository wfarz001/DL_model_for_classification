# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 18:53:02 2021

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
# from tensorflow import keras
# from tensorflow.keras import layers

cancer_tr=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Cancer/Resized_Images/Training')

cancer_train_list=[]

for file in cancer_tr.iterdir():
    cancer_train_list.append(file)
    
df_cancer=pd.DataFrame(cancer_train_list,columns=['Image_path'])

y_cancer=[1] * 42

y_cancer_df=pd.DataFrame(y_cancer,columns=['y_train'])

df_1=pd.concat((df_cancer,y_cancer_df),axis=1)


non_cancer_tr=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Non_Cancer/Resized_Images/Training') 

non_cancer_train=[]

  
for file in non_cancer_tr.iterdir():
    non_cancer_train.append(file)

df_non_cancer=pd.DataFrame(non_cancer_train,columns=['Image_path'])

y_non_cancer=[0] * 42

y_non_cancer_df=pd.DataFrame(y_non_cancer,columns=['y_train'])

df_2=pd.concat((df_non_cancer,y_non_cancer_df),axis=1)


df=pd.concat((df_1,df_2),axis=0)

from sklearn.utils import shuffle
df = shuffle(df)

X_train=[]
for file in df['Image_path']:
    #print(file)
    image=cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
    X_train.append(image)

X_train=np.array(X_train)
y_train=df.iloc[:,-1:]
# y_train=np.array(y_train)

from tensorflow import keras
from tensorflow.keras import layers

from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers

from keras.layers import BatchNormalization
# from keras.utils.np_utils import to_categorical

# # convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, 2)

from keras.applications.vgg16 import VGG16

VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in VGG_model.layers:
	layer.trainable = False
    
#VGG_model.summary()  #Trainable parameters will be 0
feature_extractor=VGG_model.predict(X_train)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)


train_features=pd.DataFrame(features)
train_features.to_csv("E:/Fall-2021/ML/project2/train_features_VGG16.csv")


from sklearn.svm import SVC  
clf = SVC(kernel='linear',probability=True)

clf.fit(features, y_train)


############Testing Part##############
cancer_te=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Cancer/Resized_Images/Testing')

cancer_te_list=[]

for file in cancer_te.iterdir():
    cancer_te_list.append(file)
    
df_cancer_te=pd.DataFrame(cancer_te_list,columns=['Image_path'])

y_cancer_te=[1] * 42

y_cancer_df_te=pd.DataFrame(y_cancer_te,columns=['y_test'])

df_3=pd.concat((df_cancer_te,y_cancer_df_te),axis=1)


non_cancer_te=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Non_Cancer/Resized_Images/Testing') 

non_cancer_te_list=[]

  
for file in non_cancer_te.iterdir():
    non_cancer_te_list.append(file)

df_non_cancer_te=pd.DataFrame(non_cancer_te_list,columns=['Image_path'])

y_non_cancer_te=[0] * 162

y_non_cancer_df_te=pd.DataFrame(y_non_cancer_te,columns=['y_test'])

df_4=pd.concat((df_non_cancer_te,y_non_cancer_df_te),axis=1)


df_te=pd.concat((df_3,df_4),axis=0)

from sklearn.utils import shuffle
df_te = shuffle(df_te)

X_test=[]
for file in df_te['Image_path']:
    #print(file)
    image=cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
    X_test.append(image)

X_test=np.array(X_test)
y_test=df_te.iloc[:,-1:]
# y_test=np.array(y_test)


X_test_feature = VGG_model.predict(X_test)
X_test_features = X_test_feature.reshape(X_test_feature.shape[0], -1)

test_features=pd.DataFrame(X_test_features)
test_features.to_csv("E:/Fall-2021/ML/project2/test_features_VGG16.csv")

pred=clf.predict(X_test_features)
pred_proba=clf.predict_proba(X_test_features)[::,1]

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, pred))

from sklearn.metrics import roc_auc_score
roc_score=roc_auc_score(y_test,pred)

print("ROC_Score=",roc_score)
auc = metrics.roc_auc_score(y_test, pred_proba)

from sklearn.metrics import classification_report

report=classification_report(y_test,pred)
print(report)

from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y_test,  pred_proba)
import matplotlib.pyplot as plt
#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()