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
import matplotlib.pyplot as plt
from fish_functions import proc_img
import fish_functions as ff


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

image_size = (224, 224)
batch_size = 32


#---------------------------------- MODEL BUILD -------------------------------

#in this basic model build from scratch we will use the image_dataset_from_directory keras function, which will create our tf.data.dataset for us without the need to label our data. Labels are assumed from the directory they are in which is consistent with our data structure. The labels are label encoded with an integer id, not one hot encoding. It also organises our data into batches.

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size)

#takes the first batch in the dataset
train_batch = train_ds.take(1)

#inspecting the object it is formed of two objects of shape ((None, 224, 224, 3), (None,))
train_batch

#accessing the two 'objects' in the dataset can be achieved with the in statement
plt.figure(figsize=(16, 8))
for images, labels in train_batch:
    for i in range(32):
        ax = plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')