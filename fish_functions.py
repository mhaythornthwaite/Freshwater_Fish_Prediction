# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:46:30 2020

@author: mhayt
"""


import os
from PIL import Image
import random
import string
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow import keras


def smooth_filter(y, box_pts, set_nan=True):
    '''
    smooths a 1d array with a convultional filter

    Parameters
    ----------
    y : numpy array 
        1d array of float or integers to be smoothed.
    box_pts : int
        size of convultional filter.
    set_nan : bool, optional
        removes edge effects by replacing values with NaN. The default is True.

    Returns
    -------
    y_smooth : numpy array
        1d smoothed array.

    '''
    
    if box_pts % 2 == 0:
        box_pts = box_pts + 1
        
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    edge = int((box_pts - 1) / 2)
    
    if set_nan:
        y_smooth[:edge] = ['NaN'] * edge
        y_smooth[-edge:] = ['NaN'] * edge
    else:
        y_smooth[:edge] = y[:edge]
        y_smooth[-edge:] = y[-edge:]        
    
    return y_smooth



def reset_weights(model):
    '''
    Re-intantiates a keras model with random weights

    Parameters
    ----------
    model : tf.keras.model
        Keras model of dense or convolutional layers. Not tested on other layer types

    '''
    
    for layer in model.layers: 
        if isinstance(layer, tf.keras.Model):
            reset_weights(layer)
            continue
        for k, initializer in layer.__dict__.items():
            if "initializer" not in k:
                continue
            #find the corresponding variable
            var = getattr(layer, k.replace("_initializer", ""))
            var.assign(initializer(var.shape, var.dtype))


def get_random_string(length):
    '''
    creates random string of characters of defined length.

    Parameters
    ----------
    length : int
        length of the random string.

    Returns
    -------
    result_str : string
        random string of characters.

    '''
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def img_to_jpg(input_path, input_dir):
    '''
    converts all files within a directory to .jpg.

    Parameters
    ----------
    input_path : string
        absolute path to the directory of the data.
    input_dir : string
        additional optional sub-directory - useful for iterating over directories.

    '''
    
    input_full = input_path + input_dir
    dir_items = os.listdir(input_full)
    
    #converting to .jpg if not already and removes the pre-converted file.
    for i, img in enumerate(dir_items):
        im_name = os.path.splitext(img)[0]
        im_type = os.path.splitext(img)[1]
        if im_type != '.jpg':
            im = Image.open(f'{input_full}/{img}')
            rgb_im = im.convert('RGB')
            rgb_im.save(f'{input_full}/{im_name}.jpg')
            im.close()
            os.remove(f'{input_full}/{img}')
 
        
def check_rgb(input_path, input_dir):
    '''
    checks that the jpg is formatted as an RBG by checking the number of channels is equal to 3. If not the data is deleted

    Parameters
    ----------
    input_path : string
        absolute path to the directory of the data.
    input_dir : string
        additional optional sub-directory - useful for iterating over directories.
        
    '''
    
    input_full = input_path + input_dir
    dir_items = os.listdir(input_full)
    
    for i, img in enumerate(dir_items):
        im = Image.open(f'{input_full}/{img}')
        im = np.asarray(im)
        im_shape = im.shape
        im_channels = im_shape[-1]
        
        if im_channels != 3:
            os.remove(f'{input_full}/{img}')
    
        
    
def rename_files(fish_species, input_path, input_dir, file_already_exisits=True):
    '''
    renames alls files in a directory with the name of the fish_species

    Parameters
    ----------
    fish_species : string
        fish species / type used in the naming convention
    input_path : string
        absolute path to the directory of the data.
    input_dir : string
        additional optional sub-directory - useful for iterating over directories.

    '''
    input_full = input_path + input_dir
    dir_items = os.listdir(input_full)
    
    if file_already_exisits:
        for i, img in enumerate(dir_items):
            new_name = get_random_string(20) + str(i) + '.jpg'
            src = input_full + '/' + img
            dst = input_full + '/' + new_name
            if src == dst:
                continue
            os.rename(src, dst)        
    
    dir_items = os.listdir(input_full)
    
    for i, img in enumerate(dir_items):
        new_name = fish_species + '_' + str(i) + '.jpg'
        src = input_full + '/' + img
        dst = input_full + '/' + new_name
        if src == dst:
            continue
        os.rename(src, dst)


def open_jpeg_as_np(path, image_size, vectorize=True):
    '''
    takes the path of a jpeg file, re-sizes, converts to grayscale and outputs a np array

    Parameters
    ----------
    path : string
        path to the image.
    image_size : tuple
        2D-tuple: (width, height).
    vectorize : TYPE, optional
        output n array will be vector. The default is True.        

    Returns
    -------
    im : array
        np array of the input image.

    '''
    
    im = Image.open(path)
    im = im.resize(image_size)
    if vectorize:
        im = im.convert('L')
    im = np.asarray(im)
    
    return im


def gen_data_array_vector(label_paths, image_size):
    '''
    takes labels from a list, loads to np array, reshaes to image_size, converts to vector and appends toa data array. 

    Parameters
    ----------
    label_paths : list
        relative paths to the data.
    image_size : tuple
        2D-tuple: (width, height).

    Returns
    -------
    data_array : np array
        data loaded to a np array.

    '''
    
    #creating required variables.
    image_vector_len = image_size[0] * image_size[1]
    num_images = len(label_paths)
    
    #instantiating our np array with zeros, speeds up writing data to this array in the for loop.
    data_array = np.zeros(shape=(num_images, image_vector_len))

    for i, path in enumerate(label_paths):
        im = open_jpeg_as_np(path, image_size)
        im_vector = np.reshape(im, image_vector_len)
        data_array[i] = im_vector
        
    return data_array


def gen_data_array_image(label_paths, image_size, RGB=True):
    '''
    takes labels from a list, loads to np array, reshapes to image_size and appends toa data array. 

    Parameters
    ----------
    label_paths : list
        relative paths to the data.
    image_size : tuple
        2D-tuple: (width, height).
    RGB : bool, optional
        Number of channels set to 3 when rbg is set to true. The default is True.

    Returns
    -------
    data_array : np array
        data loaded to a np array.

    '''
    
    #creating required variables.
    num_images = len(label_paths) 
    
    if RGB:
        data_array_shape = (num_images, image_size[1], image_size[0], 3)
    else:
        data_array_shape = (num_images,) + image_size
    
    #instantiating our np array with zeros, speeds up writing data to this array in the for loop.
    data_array = np.zeros(shape=data_array_shape)

    for i, path in enumerate(label_paths):
        im = open_jpeg_as_np(path, image_size, vectorize=False)
        data_array[i] = im
        
    return data_array


def proc_img(im_path, img_size=224):
    '''
    Takes an image path, reads the image, converts to a tensor (dtype), resizes to img_size*img_size, before returning the 'image' sensor.

    Parameters
    ----------
    im_path : str
        path to the image.
    img_size : int, optional
        width and height of the output tensor. The default is 224.

    Returns
    -------
    im : tensor
        reformated image as tensor in standard size.

    '''
    
    #loading image to variable
    im = tf.io.read_file(im_path)
    
    #modify im variable to tensor with 3 channels (RGB)
    im = tf.image.decode_jpeg(im, channels=3)
    
    #feature scaling - we're using normalisation (0 -> 1) but we could use standardisation (mean = 0, var = 1) 
    im = tf.image.convert_image_dtype(im, tf.float32)
    
    #resize the image - all images will be the same size and hence have the same number of features (pixels)
    im = tf.image.resize(im, size=[img_size, img_size])
    
    return im


def get_methods(object, spacing=20):
    '''
    Prints the methods available for a definted instantiated object

    Parameters
    ----------
    object : any
        object for inspection.
    spacing : int, optional
        print spacing. The default is 20.

    Returns
    -------
    print to the console.

    '''
  
    methodList = []
    for method_name in dir(object):
      try:
          if callable(getattr(object, method_name)):
              methodList.append(str(method_name))
      except:
          methodList.append(str(method_name))
    processFunc = (lambda s: ' '.join(s.split())) or (lambda s: s)
    for method in methodList:
      try:
          print(str(method.ljust(spacing)) + ' ' +
                processFunc(str(getattr(object, method).__doc__)[0:90]))
      except:
          print(method.ljust(spacing) + ' ' + ' getattr() failed')


def print_tf_setup():
    '''
    Prints information on the tf setup to the console.

    Returns
    -------
    None.

    '''
    
    print(' ---------------------------------------\n                 TF SETUP\n')
    physical_devices = tf.config.list_physical_devices('GPU')
    print('TF Version:', tf.__version__, '\nTF Hub Version:', hub.__version__, '\n')
    print(f'{len(physical_devices)} GPU is available' if physical_devices else 'GPU is not available')
    print(' ---------------------------------------\n')
    
    return None


def n_retraining(model, n, train_data, train_labels, val_data, val_labels, smooth=True, s=5, epochs=100, batch_size=32): 
    '''
    Retrains a model n times, randomly instantiating the model with weights on each iteration to attain more robust model performance metrics

    Parameters
    ----------
    model : tf.keras.model
        Keras model of dense or convolutional layers. Not tested on other layer types
    n : int
        number of iterations to retrain the model. More iterations will attain a more stable result but will be more compuatationally expensive.
    train_data : numpy array
        array of training data.
    train_labels : numpy array
        array of training data labels.
    val_data : numpy array
        array of validation data.
    val_labels : numpy array
        array of validation data labels.
    smooth : bool optional
        Smooth the metrics. The default is True.
    s : int, optional
        filter length of smoothing operator. The default is 5.
    epochs : int, optional
        number of epochs for model training. The default is 100.
    batch_size : int, optional
        batch size per iteration. The default is 32.

    Returns
    -------
    metrics_dict : dictionary 
        training and validation loss and accuracy metrics (mean and standard deviation).

    '''
    
    train_loss_array = np.empty((0,epochs), float)
    val_loss_array = np.empty((0,epochs), float)

    train_accuracy_array = np.empty((0,epochs), float)
    val_accuracy_array = np.empty((0,epochs), float)

    for _ in range(n):
        
        reset_weights(model)

        clf = model.fit(train_data, 
                               train_labels, 
                               epochs=epochs, 
                               batch_size=batch_size,
                               validation_data=(val_data, val_labels))
    
        history_dict = clf.history

        train_loss_list = history_dict['loss']
        val_loss_list = history_dict['val_loss']
        train_accuracy_list = history_dict['accuracy']
        val_accuracy_list = history_dict['val_accuracy']
        
        if smooth:
            train_loss_list = smooth_filter(train_loss_list, s)
            val_loss_list = smooth_filter(val_loss_list, s)
            train_accuracy_list = smooth_filter(train_accuracy_list, s)
            val_accuracy_list = smooth_filter(val_accuracy_list, s)

        train_loss_array = np.append(train_loss_array, np.array([train_loss_list]), axis=0)
        val_loss_array = np.append(val_loss_array, np.array([val_loss_list]), axis=0)
        train_accuracy_array = np.append(train_accuracy_array, np.array([train_accuracy_list]), axis=0)
        val_accuracy_array = np.append(val_accuracy_array, np.array([val_accuracy_list]), axis=0)

    metrics_dict = {
        'train_loss_mean': np.mean(train_loss_array, 0),
        'train_loss_std_p': np.mean(train_loss_array, 0) + np.std(train_loss_array, 0),
        'train_loss_std_n': np.mean(train_loss_array, 0) - np.std(train_loss_array, 0),
        'val_loss_mean': np.mean(val_loss_array, 0),
        'val_loss_std_p': np.mean(val_loss_array, 0) + np.std(val_loss_array, 0),
        'val_loss_std_n': np.mean(val_loss_array, 0) - np.std(val_loss_array, 0),
        'train_acc_mean': np.mean(train_accuracy_array, 0),
        'train_acc_std_p': np.mean(train_accuracy_array, 0) + np.std(train_accuracy_array, 0),
        'train_acc_std_n': np.mean(train_accuracy_array, 0) - np.std(train_accuracy_array, 0),
        'val_acc_mean': np.mean(val_accuracy_array, 0),
        'val_acc_std_p': np.mean(val_accuracy_array, 0) + np.std(val_accuracy_array, 0),
        'val_acc_std_n': np.mean(val_accuracy_array, 0) - np.std(val_accuracy_array, 0)
        }    
        
    return metrics_dict


def n_retraining_datagen(model, n, train_generator, val_generator, smooth=True, s=5, epochs=100, batch_size=32): 
    '''
    Retrains a model n times, randomly instantiating the model with weights on each iteration to attain more robust model performance metrics

    Parameters
    ----------
    model : tf.keras.model
        Keras model of dense or convolutional layers. Not tested on other layer types
    n : int
        number of iterations to retrain the model. More iterations will attain a more stable result but will be more compuatationally expensive.
    train_generator : keras.preprocessing.image.DirectoryIterator
        training data generator.
    val_generator : keras.preprocessing.image.DirectoryIterator
        validation data generator.
    smooth : bool optional
        Smooth the metrics. The default is True.
    s : int, optional
        filter length of smoothing operator. The default is 5.
    epochs : int, optional
        number of epochs for model training. The default is 100.
    batch_size : int, optional
        batch size per iteration. The default is 32.

    Returns
    -------
    metrics_dict : dictionary 
        training and validation loss and accuracy metrics (mean and standard deviation).

    '''
    
    train_loss_array = np.empty((0,epochs), float)
    val_loss_array = np.empty((0,epochs), float)

    train_accuracy_array = np.empty((0,epochs), float)
    val_accuracy_array = np.empty((0,epochs), float)

    for _ in range(n):
        
        reset_weights(model)
        
        steps_per_train_epoch = train_generator.__len__()
        steps_per_val_epoch = val_generator.__len__()
        
        clf = model.fit(train_generator,
                        steps_per_epoch=steps_per_train_epoch,
                        epochs=epochs,
                        validation_data=val_generator,
                        validation_steps=steps_per_val_epoch)


    
        history_dict = clf.history

        train_loss_list = history_dict['loss']
        val_loss_list = history_dict['val_loss']
        train_accuracy_list = history_dict['accuracy']
        val_accuracy_list = history_dict['val_accuracy']
        
        if smooth:
            train_loss_list = smooth_filter(train_loss_list, s)
            val_loss_list = smooth_filter(val_loss_list, s)
            train_accuracy_list = smooth_filter(train_accuracy_list, s)
            val_accuracy_list = smooth_filter(val_accuracy_list, s)

        train_loss_array = np.append(train_loss_array, np.array([train_loss_list]), axis=0)
        val_loss_array = np.append(val_loss_array, np.array([val_loss_list]), axis=0)
        train_accuracy_array = np.append(train_accuracy_array, np.array([train_accuracy_list]), axis=0)
        val_accuracy_array = np.append(val_accuracy_array, np.array([val_accuracy_list]), axis=0)

    metrics_dict = {
        'train_loss_mean': np.mean(train_loss_array, 0),
        'train_loss_std_p': np.mean(train_loss_array, 0) + np.std(train_loss_array, 0),
        'train_loss_std_n': np.mean(train_loss_array, 0) - np.std(train_loss_array, 0),
        'val_loss_mean': np.mean(val_loss_array, 0),
        'val_loss_std_p': np.mean(val_loss_array, 0) + np.std(val_loss_array, 0),
        'val_loss_std_n': np.mean(val_loss_array, 0) - np.std(val_loss_array, 0),
        'train_acc_mean': np.mean(train_accuracy_array, 0),
        'train_acc_std_p': np.mean(train_accuracy_array, 0) + np.std(train_accuracy_array, 0),
        'train_acc_std_n': np.mean(train_accuracy_array, 0) - np.std(train_accuracy_array, 0),
        'val_acc_mean': np.mean(val_accuracy_array, 0),
        'val_acc_std_p': np.mean(val_accuracy_array, 0) + np.std(val_accuracy_array, 0),
        'val_acc_std_n': np.mean(val_accuracy_array, 0) - np.std(val_accuracy_array, 0)
        }    
        
    return metrics_dict

