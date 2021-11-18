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
from tensorflow.keras.applications import VGG16, Xception
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from fish_functions import print_tf_setup, open_jpeg_as_np, gen_data_array_image, n_retraining_datagen, smooth_filter
import copy

plt.close('all')

#note that to allow tensorflow to be compatible with the GPU cudnn==7.6.4 was installed in the environment. 
print_tf_setup()


#------------------------------- INPUT VARIABLES ------------------------------

image_size = (299, 299)
input_shape = image_size + (3,)
batch_size = 32
num_classes = 14
num_epochs = 5
transfer_model = Xception #select from here: https://keras.io/api/applications/

train_dir = 'data_for_generator/train_data'
test_dir = 'data_for_generator/test_data'


#-------------------------------- DATA GENERATOR ------------------------------

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
    t = data_batch
    t2 = labels_batch
    break

steps_per_train_epoch = train_generator.__len__()
steps_per_val_epoch = test_generator.__len__()


#---------------------------------- MODEL BUILD -------------------------------

#here we have two different options, we can run the convolutional base over the training images and use hese as direct inputs to a dense classification network. This may be reffered to as fast feature extraction and is computationally cheap. Or we can set the conv base as non-trainable adding a dense classifier on top, simply called feature extraction. This technique allows for data augmentation in the training stage but we have to run the entire dataset through the conv base for every epoch, therefore is much slower and more computationally exensive. Since we have a GPU we're going with option two to benefit from data augmentation.

conv_base = transfer_model(weights='imagenet',
                           include_top=False,
                           input_shape=input_shape)

conv_base.summary()

model = keras.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(14, activation='softmax'))

conv_base.trainable = False
model.summary()

optimiser = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=optimiser,
              metrics=["accuracy"])

clf = model.fit(train_generator,
                steps_per_epoch=steps_per_train_epoch,
                epochs=2,
                validation_data=test_generator,
                validation_steps=steps_per_val_epoch)

metrics_dict = n_retraining_datagen(model=model, 
                                    n=2, 
                                    train_generator=train_generator,
                                    val_generator=test_generator,
                                    epochs=num_epochs,
                                    batch_size=batch_size,
                                    transfer_model=True)

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
fig.suptitle('Training & Validation Loss Xception Transfer Model', y=0.95, fontsize=14, fontweight='bold')

#plotting training and validation loss
ax.plot(epochs, metrics_dict_smooth['train_loss_mean'], 'b', label='Training Loss')
ax.plot(epochs, metrics_dict_smooth['val_loss_mean'], 'r', label='Validation Loss')
ax.plot(epochs, metrics_dict_smooth['val_loss_std_p'], label='_nolegend_', alpha=0)
ax.plot(epochs, metrics_dict_smooth['val_loss_std_n'], label='_nolegend_', alpha=0)
ax.fill_between(epochs, metrics_dict_smooth['val_loss_std_p'], metrics_dict_smooth['val_loss_std_n'], color='grey', alpha=0.15)
ax.axhline(np.nanmin(metrics_dict_smooth['val_loss_mean']), c='r', alpha=0.3, ls='dashed', label='Min Validation Loss')

#setting axis limits, labels and legend
ax.set_ylim([np.nanmin(metrics_dict_smooth['train_loss_mean'])-0.25, np.nanmax(metrics_dict_smooth['val_loss_mean'])+1.75])
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()

#--------- TRAINING & VALIDATION ACCURACY ---------

#fig setup
fig2, ax = plt.subplots()
fig2.suptitle('Training & Validation Accuracy Xception Transfer Model', y=0.95, fontsize=14, fontweight='bold')

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

