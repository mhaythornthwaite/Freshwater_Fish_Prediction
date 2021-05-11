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
from fish_functions import print_tf_setup, open_jpeg_as_np, gen_data_array

plt.close('all')

#note that to allow tensorflow to be compatible with the GPU cudnn==7.6.4 was installed in the environment. 
print_tf_setup()


#--------------------------------- DATA LOADING -------------------------------

with open('data_labels/one_hot_labels', 'rb') as myFile:
    one_hot_labels = pickle.load(myFile)

with open('data_labels/labels', 'rb') as myFile:
    labels = pickle.load(myFile)
    
with open('data_labels/label_paths', 'rb') as myFile:
    label_paths = pickle.load(myFile)
    
    
#------------------------------- INPUT VARIABLES ------------------------------

image_size = (64, 64)
batch_size = 32
num_classes = 14

image_vector_len = image_size[0] * image_size[1]
num_images = len(labels)


#----------------------------------- DATA PREP --------------------------------

#in this basic model build from scratch we need to write some functions that will load our data into a numpy array. As we are begining with a basic model build we will need to convert our 2D image matrics into 1D image vectors. We have variable images sizes which will need to be normalised to a single size, and likely downsampled in most cases as we only have around 1000 samples. The size of the input image will be tested as a hyperparameter. 

#opening a test image and plotting to see if the class recognisable with the current downsampling
im = open_jpeg_as_np(label_paths[0], image_size)
plt.imshow(im, cmap='gray', vmin=0, vmax=255)

#loading all our data to a np array, and train test split
data_array = gen_data_array(label_paths, image_size)
train_images, test_images, train_labels, test_labels = train_test_split(data_array, one_hot_labels, test_size=0.2)

#train_images = train_images[:100]
#train_labels = train_labels[:100]

#---------------------------------- MODEL BUILD -------------------------------

simple_model = keras.Sequential([
    layers.Dense(192, activation='relu', name='layer1', input_shape=(image_vector_len,)),
    layers.Dense(48, activation='relu', name='layer2'),
    layers.Dense(14, activation='softmax', name='layer3'),
    ])

#full list of keras optimisers: https://keras.io/api/optimizers/
#low learning rate: https://stackoverflow.com/questions/41488279/neural-network-always-predicts-the-same-class
optimiser = keras.optimizers.Adam(learning_rate=0.00001)
optimiser = keras.optimizers.RMSprop(learning_rate=0.00001)

simple_model.compile(loss='categorical_crossentropy',
                     optimizer=optimiser,
                     metrics=["accuracy"]
                     )

simple_model.summary()

history = simple_model.fit(train_images, 
                           train_labels, 
                           epochs=1000, 
                           batch_size=batch_size,
                           validation_data=(test_images, test_labels))


#------------------------------- MODEL PERFORMANCE ----------------------------

#--------- TRAINING & VALIDATION LOSS ---------

#setting up plottable variables
history_dict = history.history
loss_values = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = list(range(1, len(loss_values)+1))

#fig setup including twin axis
fig, ax = plt.subplots()
fig.suptitle('Training & Validation Loss', y=0.95, fontsize=16, fontweight='bold')
ax2 = ax.twinx()

#plotting training and validation loss
train_loss = ax.plot(epochs, loss_values, 'r', label='Training Loss')
val_loss = ax2.plot(epochs, val_loss, 'b', label='Validation Loss')

#setting axis limits
ax.set_ylim([min(loss_values)-0.2, 2.8])
ax2.set_ylim([min(val_loss)-1, min(val_loss)+10])

#plotting legend
lns = train_loss + val_loss
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0)

train_predictions = simple_model.predict(train_images[:100])



#https://www.youtube.com/watch?v=J6Ok8p463C4


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')