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
from tensorflow import keras
from tensorflow.keras import layers
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


#----------------------------------- DATA PREP --------------------------------

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

#the above functions create a label encoded result of (img, id). The id is simply the order of the folders presented. Therefore we can create a {class: id} dictionary so we understand the meaning of the id.
id_class = dict(zip(list(range(14)), os.listdir('data')))
class_id = dict(zip(os.listdir('data'), list(range(14))))

#takes the first batch in the dataset
train_batch = train_ds.take(1)

#inspecting the object it is formed of two objects of shape ((None, 224, 224, 3), (None,))
train_batch


#---------- BATCH VISUALISATION ----------

#accessing the two 'objects' in the dataset can be achieved with the in statement
fig = plt.figure(figsize=(16, 8))
fig.suptitle('All Images in a Single Batch', y=0.955, fontsize=16, fontweight='bold');
for images, labels in train_batch:
    for i in range(32):
        ax = plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_class = id_class[int(labels[i])]
        plt.title(label_class)
        plt.axis("off")


#---------- DATA AUGMENTATION ----------

data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1)])

fig2 = plt.figure(figsize=(10, 10))
fig2.suptitle('Data Augmentation on a Single Image', y=0.94, fontsize=16, fontweight='bold');
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")





# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')