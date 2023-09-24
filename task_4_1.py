# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 17:53:23 2021

@author: Windows
"""

import pandas as pd
import pathlib
import glob
import os
from PIL import Image
import SimpleITK as sitk
import cv2
from os.path import join

img_path=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Non_Cancer/Testing')
out_path=pathlib.Path('E:/Fall-2021/ML/project2/Skin_Data/Non_Cancer/Resized_Images/Testing/')

# for folder in img_path.iterdir():
#     if folder.is_dir():
#         print(folder.name)
for file in img_path.iterdir():
        image=cv2.imread(str(file),cv2.IMREAD_UNCHANGED)
        img=cv2.resize(image,(224,224),interpolation = cv2.INTER_AREA)
        print(file.name)
        cv2.imwrite(join(out_path,file.name), img)