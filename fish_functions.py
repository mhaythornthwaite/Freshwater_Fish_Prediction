# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:46:30 2020

@author: mhayt
"""


import os
from PIL import Image
import random
import string


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
    
