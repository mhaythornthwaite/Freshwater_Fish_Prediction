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
from tensorflow.keras.applications import VGG16
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from fish_functions import print_tf_setup, open_jpeg_as_np, gen_data_array_image

plt.close('all')

#note that to allow tensorflow to be compatible with the GPU cudnn==7.6.4 was installed in the environment. 
print_tf_setup()


#------------------------------- INPUT VARIABLES ------------------------------

image_size = (64, 64)
input_shape = image_size + (3,)
batch_size = 32
num_classes = 14

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

conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=input_shape)

conv_base.summary()

model = keras.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(14, activation='softmax'))

conv_base.trainable = False
model.summary()

optimiser = keras.optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimiser,
              metrics=["accuracy"])

clf = model.fit_generator(train_generator,
                          steps_per_epoch=steps_per_train_epoch,
                          epochs=10,
                          validation_data=test_generator,
                          validation_steps=steps_per_val_epoch)


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')

