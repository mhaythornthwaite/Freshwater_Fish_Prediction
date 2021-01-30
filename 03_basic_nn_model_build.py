# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 12:54:53 2021

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#------------------------------ FISH PREDICTION ------------------------------

import time
start=time.time()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
from fish_functions import proc_img


#-------------------------- CHECKING TF VERSION AND GPU -----------------------

#note that to allow tensorflow to be compatible with the GPU cudnn==7.6.4 was installed in the environment. 

print(' ---------------------------------------\n                 TF SETUP\n')
physical_devices = tf.config.list_physical_devices('GPU')
print('TF Version:', tf.__version__, '\nTF Hub Version:', hub.__version__, '\n')
print(f'{len(physical_devices)} GPU is available' if physical_devices else 'GPU is not available')
print(' ---------------------------------------\n')


#--------------------------------- DATA LOADING -------------------------------

with open('data_labels/one_hot_labels', 'rb') as myFile:
    one_hot_labels = pickle.load(myFile)

with open('data_labels/labels', 'rb') as myFile:
    labels = pickle.load(myFile)
    
with open('data_labels/label_paths', 'rb') as myFile:
    label_paths = pickle.load(myFile)
    
    
#------------------------------- INPUT VARIABLES ------------------------------

#number of images gone into our training + validation set, set to 1011 to use full dataset
num_images = 500


#------------------------------- DATA PREPARATION -----------------------------
  
X = label_paths
y = one_hot_labels

#experimenting will start off with a largely trimmed version of the dataset, ~500 images instead of 1011. This will speed up the inital testing/experimentation

X_train, X_val, y_train, y_val = train_test_split(X[:num_images], 
                                                  y[:num_images], 
                                                  test_size = 0.2, 
                                                  random_state = 42)


test = proc_img(label_paths[0])
test.shape
t2 = test.numpy()


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')