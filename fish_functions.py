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
    