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

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
from fish_functions import print_tf_setup, open_jpeg_as_np, gen_data_array_vector, n_retraining

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
batch_size = 32
num_classes = 14
num_epochs = 100

image_vector_len = image_size[0] * image_size[1]
num_images = len(labels)


#----------------------------------- DATA PREP --------------------------------

#in this basic model build from scratch we need to write some functions that will load our data into a numpy array. As we are begining with a basic model build we will need to convert our 2D image matrics into 1D image vectors. We have variable images sizes which will need to be normalised to a single size, and likely downsampled in most cases as we only have around 1000 samples. The size of the input image will be tested as a hyperparameter. 

#opening a test image and plotting to see if the class recognisable with the current downsampling
im = open_jpeg_as_np(label_paths[0], image_size)
plt.imshow(im, cmap='gray', vmin=0, vmax=255)

#loading all our data to a np array, and train test split
data_array = gen_data_array_vector(label_paths, image_size)
train_images, test_images, train_labels, test_labels = train_test_split(data_array, one_hot_labels, test_size=0.2)


#---------------------------------- MODEL BUILD -------------------------------

simple_model = keras.Sequential([
    layers.Dense(192, activation='relu', name='layer1', input_shape=(image_vector_len,)),
    layers.Dense(96, activation='relu', name='layer2'),
    layers.Dense(48, activation='relu', name='layer3'),
    layers.Dense(14, activation='softmax', name='layer4'),
    ])

#full list of keras optimisers: https://keras.io/api/optimizers/
#low learning rate: https://stackoverflow.com/questions/41488279/neural-network-always-predicts-the-same-class
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
#optimiser = keras.optimizers.RMSprop(learning_rate=0.00001)

simple_model.compile(loss='categorical_crossentropy',
                     optimizer=optimiser,
                     metrics=["accuracy"]
                     )

simple_model.summary()

clf = simple_model.fit(train_images, 
                       train_labels, 
                       epochs=num_epochs, 
                       batch_size=batch_size,
                       validation_data=(test_images, test_labels))

metrics_dict = n_retraining(model=simple_model, 
                            n=10, 
                            train_data=train_images, 
                            train_labels=train_labels, 
                            val_data=test_images,
                            val_labels=test_labels,
                            epochs=num_epochs,
                            batch_size=batch_size,
                            s=3)

#looking at predictions on a collection of test and train images for basic inspection and QC.
train_predictions = simple_model.predict(train_images[:100])
val_predictions = simple_model.predict(test_images[:100])


#------------------------------- MODEL PERFORMANCE ----------------------------

#--------- TRAINING & VALIDATION LOSS ---------

#setting up plottable variables
history_dict = clf.history
loss_values = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = list(range(1, len(loss_values)+1))

#fig setup including twin axis
fig, ax = plt.subplots()
fig.suptitle('Training & Validation Loss Basic Dense Network', y=0.95, fontsize=14, fontweight='bold')

#plotting training and validation loss
ax.plot(epochs, metrics_dict['train_loss_mean'], 'b', label='Training Loss')
ax.plot(epochs, metrics_dict['val_loss_mean'], 'r', label='Validation Loss')
ax.plot(epochs, metrics_dict['val_loss_std_p'], label='_nolegend_', alpha=0)
ax.plot(epochs, metrics_dict['val_loss_std_n'], label='_nolegend_', alpha=0)
ax.fill_between(epochs, metrics_dict['val_loss_std_p'], metrics_dict['val_loss_std_n'], color='grey', alpha=0.15)
ax.axhline(np.nanmin(metrics_dict['val_loss_mean']), c='r', alpha=0.3, ls='dashed', label='Min Validation Loss')

#setting axis limits, labels and legend
ax.set_ylim([np.nanmin(metrics_dict['train_loss_mean'])-0.5, np.nanmin(metrics_dict['train_loss_mean'])+4])
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()

#--------- TRAINING & VALIDATION ACCURACY ---------

#setting up plottable variables
accuracy_values = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

#fig setup
fig2, ax = plt.subplots()
fig2.suptitle('Training & Validation Accuracy Basic Dense Network', y=0.95, fontsize=14, fontweight='bold')

#plotting training and validation accuracy
ax.plot(epochs, metrics_dict['train_acc_mean'], 'b', label='Training Accuracy')
ax.plot(epochs, metrics_dict['val_acc_mean'], 'r', label='Validation Accuracy')
ax.plot(epochs, metrics_dict['val_acc_std_p'], label='_nolegend_', alpha=0)
ax.plot(epochs, metrics_dict['val_acc_std_n'], label='_nolegend_', alpha=0)
ax.fill_between(epochs, metrics_dict['val_acc_std_p'], metrics_dict['val_acc_std_n'], color='grey', alpha=0.15)

#plotting accuracy lines
ax.axhline(np.nanmax(metrics_dict['val_acc_mean']), c='r', alpha=0.3, ls='dashed', label='Max Validation Accuracy')
ax.axhline(1/14, c='k', alpha=0.3, ls='dashed', label='Random Guess Accuracy')

#plotting legend and setting limits
ax.legend()
ax.set(xlabel='Epochs',
       ylabel='Accuracy');
ax.set_ylim([0,np.nanmax(metrics_dict['train_acc_mean'])+0.1])


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')