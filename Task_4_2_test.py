# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 19:16:56 2021

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

cancer_tr=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Cancer/Resized_Images/Testing')

cancer_train_list=[]

for file in cancer_tr.iterdir():
    cancer_train_list.append(file)
    
df_cancer=pd.DataFrame(cancer_train_list,columns=['Image_path'])

y_cancer=[1] * 42

y_cancer_df=pd.DataFrame(y_cancer,columns=['y_train'])

df_1=pd.concat((df_cancer,y_cancer_df),axis=1)


non_cancer_tr=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Non_Cancer/Resized_Images/Testing') 

non_cancer_train=[]

  
for file in non_cancer_tr.iterdir():
    non_cancer_train.append(file)

df_non_cancer=pd.DataFrame(non_cancer_train,columns=['Image_path'])

y_non_cancer=[0] * 162

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

X_test=np.array(X_train)
y_test=df.iloc[:,-1:]
y_test=np.array(y_test)









model=keras.models.load_model("E:/Fall-2021/ML/project2/Task_4_VGG_model.h5")

pred = model.predict(X_test)

y_pred = np.argmax(pred, axis=1)

# y_proba=pred.argmax(axis=-1)

from sklearn.metrics import classification_report

report=classification_report(y_test,y_pred)
print(report)

###Note that in binary classification, recall of the positive class is also known as “sensitivity”; 
####recall of the negative class is “specificity”.

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
roc_score=roc_auc_score(y_test,y_pred)

from sklearn.metrics import accuracy_score
acc=accuracy_score(y_test,y_pred)

from sklearn import metrics
from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred)
auc = metrics.roc_auc_score(y_test,  y_pred)

import matplotlib.pyplot as plt
#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()