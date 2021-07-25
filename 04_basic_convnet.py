# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 19:33:44 2021

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#------------------------------ FISH PREDICTION ------------------------------

import time
start=time.time()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from fish_functions import print_tf_setup, open_jpeg_as_np, gen_data_array_image

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

image_size = (32, 32)
input_shape = image_size + (3,)
batch_size = 32
num_classes = 14

image_vector_len = image_size[0] * image_size[1]
num_images = len(labels)


#----------------------------------- DATA PREP --------------------------------

#in this basic convolutional model build from scratch we need to write some functions that will load our data into a numpy array. We have variable images sizes which will need to be normalised to a single size, and likely downsampled in most cases as we only have around 1000 samples. The size of the input image will be tested as a hyperparameter. We will retain the 3 colour channels so our data array will be as follows (#samples, #x pixels, #y pixels, #channels)

#opening a test image and plotting to see if the class recognisable with the current downsampling.
im = open_jpeg_as_np(label_paths[0], image_size, vectorize=False)
plt.imshow(im, vmin=0, vmax=255)

#loading all our data to a np array, and train test split
data_array = gen_data_array_image(label_paths, image_size)
train_images, test_images, train_labels, test_labels = train_test_split(data_array, one_hot_labels, test_size=0.2)


#---------------------------------- MODEL BUILD -------------------------------

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(14, activation='softmax')
    ])

optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimiser,
              metrics=["accuracy"])

model.summary()

clf = model.fit(train_images, 
                train_labels, 
                epochs=100, 
                batch_size=batch_size,
                validation_data=(test_images, test_labels))


#------------------------------- MODEL PERFORMANCE ----------------------------

#--------- TRAINING & VALIDATION LOSS ---------

#setting up plottable variables
history_dict = clf.history
loss_values = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = list(range(1, len(loss_values)+1))

#fig setup including twin axis
fig, ax = plt.subplots()
fig.suptitle('Training & Validation Loss Basic Convnet', y=0.95, fontsize=14, fontweight='bold')

#plotting training and validation loss
ax.plot(epochs, loss_values, 'b', label='Training Loss')
ax.plot(epochs, val_loss, 'r', label='Validation Loss')
ax.axhline(min(val_loss), c='r', alpha=0.3, ls='dashed', label='Min Validation Loss')

#setting axis limits
ax.set_ylim([min(loss_values)-0.5, min(loss_values)+8])

#plotting legend
ax.legend()

#plotting axis labels
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')

train_predictions = model.predict(train_images[:100])
val_predictions = model.predict(test_images[:100])

#--------- TRAINING & VALIDATION ACCURACY ---------

#setting up plottable variables
accuracy_values = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

#fig setup including twin axis
fig2, ax = plt.subplots()
fig2.suptitle('Training & Validation Accuracy Basic Convnet', y=0.95, fontsize=14, fontweight='bold')

#plotting training and validation loss
ax.plot(epochs, accuracy_values, 'b', label='Training Accuracy')
ax.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
ax.axhline(max(val_accuracy), c='r', alpha=0.3, ls='dashed', label='Max Validation Accuracy')
ax.axhline(1/14, c='k', alpha=0.3, ls='dashed', label='Random Guess Accuracy')

#setting axis limits
ax.set_ylim([0,max(accuracy_values)+0.1])

#plotting legend
ax.legend()
ax.set(xlabel='Epochs',
       ylabel='Accuracy');

#https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')