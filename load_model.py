# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 00:54:30 2018

@author: abhir
"""
from keras.models import model_from_json
json_file = open('catordog.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
classifier = model_from_json(loaded_model_json)

# Loading the indices to dictionary
#import pickle
#with open('indices.pickle', 'rb') as handle:
#    indices = pickle.load(handle)

# load weights into new model
classifier.load_weights("catordog.h5")
print("Loaded model from disk")

# Predicting the test set
import numpy as np
from keras.preprocessing import image
import os
for filename in os.listdir("../Section 8 - Building a CNN/dataset/predict_set"):
    im=image.load_img('../Section 8 - Building a CNN/dataset/predict_set/'+filename,target_size=(128,128))
    im=image.img_to_array(im)
    im=np.expand_dims(im,axis=0)
    result=classifier.predict(im)
    print("File name {} and its prediction is {}".format(filename,result[0][0]))
