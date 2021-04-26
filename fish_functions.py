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


def open_jpeg_as_np(path, image_size):
    '''
    takes the path of a jpeg file, re-sizes, converts to grayscale and outputs a np array

    Parameters
    ----------
    path : string
        path to the image.
    image_size : tuple
        2D-tuple: (width, height).

    Returns
    -------
    im : array
        np array of the input image.

    '''
    
    im = Image.open(path)
    im = im.resize(image_size)
    im = im.convert('L')
    im = np.asarray(im)
    
    return im


def gen_data_array(label_paths, image_size):
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