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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from fish_functions import print_tf_setup, open_jpeg_as_np, gen_data_array_image, n_retraining_datagen, smooth_filter
import copy

plt.close('all')

#note that to allow tensorflow to be compatible with the GPU cudnn==7.6.4 was installed in the environment. 
print_tf_setup()


#--------------------------------- DATA LOADING -------------------------------
    
with open('data_labels/label_paths', 'rb') as myFile:
    label_paths = pickle.load(myFile)


#------------------------------- INPUT VARIABLES ------------------------------

image_size = (64, 64)
input_shape = image_size + (3,)
batch_size = 32
num_classes = 14
num_epochs = 250

train_dir = 'data_for_generator/train_data'
test_dir = 'data_for_generator/test_data'


#-------------------------------- DATA GENERATOR ------------------------------

#note that using keras image data generators is slow. Right now this is not a major problem with this project, but going forward it may be worth considering building a proper data pipeline using tf.data
#https://www.tensorflow.org/guide/data

#augmentation to the training data
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)

#no augmentation to the test data
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=image_size,
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

test_generator = train_datagen.flow_from_directory(test_dir,
                                                   target_size=image_size,
                                                   batch_size=batch_size,
                                                   class_mode='categorical')

#checking the data and label batch shape the generator yields.
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

#checking the number of iterations per epoch = number of images / batch size
steps_per_train_epoch = train_generator.__len__()
steps_per_val_epoch = test_generator.__len__()


#----------------------- DATA AUGMENTATION VISUALISATION ----------------------

#plotting data augmentation with high resolution
im = plt.imread(label_paths[22])
im = cv2.resize(im, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(ncols=2,
                                            nrows=2,
                                            figsize=(10,12))
ax1.imshow(train_datagen.random_transform(im))
ax2.imshow(train_datagen.random_transform(im))
ax3.imshow(train_datagen.random_transform(im))
ax4.imshow(train_datagen.random_transform(im))

fig.suptitle('Data Augmentation Example: High Resolution', y=0.95, fontsize=22, fontweight='bold')


#plotting data augmentation with low resolution
im = plt.imread(label_paths[22])
im = cv2.resize(im, dsize=(64,64), interpolation=cv2.INTER_CUBIC)

fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(ncols=2,
                                            nrows=2,
                                            figsize=(10,12))
ax1.imshow(train_datagen.random_transform(im))
ax2.imshow(train_datagen.random_transform(im))
ax3.imshow(train_datagen.random_transform(im))
ax4.imshow(train_datagen.random_transform(im))

fig.suptitle('Data Augmentation Example: Low Resolution', y=0.95, fontsize=22, fontweight='bold')


#---------------------------------- MODEL BUILD -------------------------------

model = keras.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))       
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(14, activation='softmax'))

optimiser = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimiser,
              metrics=["accuracy"])

model.summary()

clf = model.fit(train_generator,
                steps_per_epoch=steps_per_train_epoch,
                epochs=2,
                validation_data=test_generator,
                validation_steps=steps_per_val_epoch)


metrics_dict = n_retraining_datagen(model=model, 
                                    n=10, 
                                    train_generator=train_generator,
                                    val_generator=test_generator,
                                    epochs=num_epochs,
                                    batch_size=batch_size)

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

epochs = list(range(1, num_epochs+1))

#fig setup including twin axis
fig, ax = plt.subplots()
fig.suptitle('Training & Validation Loss Basic CNN + Augmentation', y=0.95, fontsize=14, fontweight='bold')

#plotting training and validation loss
ax.plot(epochs, metrics_dict_smooth['train_loss_mean'], 'b', label='Training Loss')
ax.plot(epochs, metrics_dict_smooth['val_loss_mean'], 'r', label='Validation Loss')
ax.plot(epochs, metrics_dict_smooth['val_loss_std_p'], label='_nolegend_', alpha=0)
ax.plot(epochs, metrics_dict_smooth['val_loss_std_n'], label='_nolegend_', alpha=0)
ax.fill_between(epochs, metrics_dict_smooth['val_loss_std_p'], metrics_dict_smooth['val_loss_std_n'], color='grey', alpha=0.15)
ax.axhline(np.nanmin(metrics_dict_smooth['val_loss_mean']), c='r', alpha=0.3, ls='dashed', label='Min Validation Loss')

#setting axis limits, labels and legend
ax.set_ylim([np.nanmin(metrics_dict_smooth['train_loss_mean'])-0.25, np.nanmax(metrics_dict_smooth['val_loss_mean'])+0.55])
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()

#--------- TRAINING & VALIDATION ACCURACY ---------

#fig setup
fig2, ax = plt.subplots()
fig2.suptitle('Training & Validation Accuracy Basic CNN + Augmentation', y=0.95, fontsize=14, fontweight='bold')

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


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')