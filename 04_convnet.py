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
from fish_functions import print_tf_setup, open_jpeg_as_np, gen_data_array_image, n_retraining, smooth_filter
import copy

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

model = 'complex' #select 'complex' or 'simple'

if model == 'complex':
    image_size = (224, 224)
elif model == 'simple':
    image_size = (32, 32)
else:
    print('Please select either a simple or complex model')

input_shape = image_size + (3,)
batch_size = 32
num_classes = 14
num_epochs = 150

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

if model=='complex':
    model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3,3), activation='relu'),
    layers.Conv2D(256, (3,3), activation='relu'), 
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(14, activation='softmax')
    ])
elif model=='simple':
    model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
    layers.Dense(14, activation='softmax')
    ])
else:
    print('Please select either a simple or complex model')


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

metrics_dict = n_retraining(model=model, 
                            n=10, 
                            train_data=train_images, 
                            train_labels=train_labels, 
                            val_data=test_images,
                            val_labels=test_labels,
                            epochs=num_epochs,
                            batch_size=batch_size)

#looking at predictions on a collection of test and train images for basic inspection and QC.
train_predictions = model.predict(train_images[:100])
val_predictions = model.predict(test_images[:100])

#smoothing the output of the metrics dictionary ready for analysis and plotting
metrics_dict_smooth = copy.deepcopy(metrics_dict)
for key in metrics_dict_smooth:
    metrics_dict_smooth[key] = smooth_filter(metrics_dict_smooth[key], 3)

#printing validation accuracy information to the console
max_accuracy = np.nanmax(metrics_dict_smooth['val_acc_mean'])
max_accuracy_epoch = list(metrics_dict_smooth['val_acc_mean']).index(max_accuracy)
max_accuracy = round((np.nanmax(metrics_dict_smooth['val_acc_mean'])), 3) * 100
print(f'\nMax accuracy of {max_accuracy}% achieved after {max_accuracy_epoch} epochs\n')


#------------------------------- MODEL PERFORMANCE ----------------------------

#--------- TRAINING & VALIDATION LOSS ---------

#fig setup
fig, ax = plt.subplots()
fig.suptitle('Training & Validation Loss Deeper CNN', y=0.95, fontsize=14, fontweight='bold')
epochs = list(range(1, num_epochs+1))

#plotting training and validation loss
ax.plot(epochs, metrics_dict_smooth['train_loss_mean'], 'b', label='Training Loss')
ax.plot(epochs, metrics_dict_smooth['val_loss_mean'], 'r', label='Validation Loss')
ax.plot(epochs, metrics_dict_smooth['val_loss_std_p'], label='_nolegend_', alpha=0)
ax.plot(epochs, metrics_dict_smooth['val_loss_std_n'], label='_nolegend_', alpha=0)
ax.fill_between(epochs, metrics_dict_smooth['val_loss_std_p'], metrics_dict_smooth['val_loss_std_n'], color='grey', alpha=0.15)
ax.axhline(np.nanmin(metrics_dict_smooth['val_loss_mean']), c='r', alpha=0.3, ls='dashed', label='Min Validation Loss')

#setting axis limits, labels and legend
ax.set_ylim([np.nanmin(metrics_dict_smooth['train_loss_mean'])-0.5, np.nanmin(metrics_dict['train_loss_mean'])+4])
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()

#--------- TRAINING & VALIDATION ACCURACY ---------

#fig setup
fig2, ax = plt.subplots()
fig2.suptitle('Training & Validation Accuracy Deeper CNN', y=0.95, fontsize=14, fontweight='bold')

#plotting training and validation accuracy
ax.plot(epochs, metrics_dict_smooth['train_acc_mean'], 'b', label='Training Accuracy')
ax.plot(epochs, metrics_dict_smooth['val_acc_mean'], 'r', label='Validation Accuracy')
ax.plot(epochs, metrics_dict_smooth['val_acc_std_p'], label='_nolegend_', alpha=0)
ax.plot(epochs, metrics_dict_smooth['val_acc_std_n'], label='_nolegend_', alpha=0)
ax.fill_between(epochs, metrics_dict_smooth['val_acc_std_p'], metrics_dict_smooth['val_acc_std_n'], color='grey', alpha=0.15)

#plotting accuracy lines
ax.axhline(np.nanmax(metrics_dict_smooth['val_acc_mean']), c='r', alpha=0.3, ls='dashed', label='Max Validation Accuracy')
ax.axhline(1/14, c='k', alpha=0.3, ls='dashed', label='Random Guess Accuracy')

#plotting legend and setting limits
ax.legend()
ax.set(xlabel='Epochs',
       ylabel='Accuracy');
ax.set_ylim([0,np.nanmax(metrics_dict_smooth['train_acc_mean'])+0.1])

#https://stats.stackexchange.com/questions/282160/how-is-it-possible-that-validation-loss-is-increasing-while-validation-accuracy


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')