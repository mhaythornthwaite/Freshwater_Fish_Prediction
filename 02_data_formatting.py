# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:53:28 2020

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#------------------------------ FISH PREDICTION ------------------------------

import time
start=time.time()

from fish_functions import img_to_jpg, rename_files
import os
import pandas as pd
import numpy as np
import pickle


#----------------------------- DATA REFORMATTING -----------------------------

input_path = 'C:/Users/mhayt/Documents/Software_Developer-Python/Freshwater_Fish_Prediction/data/'
input_dirs = os.listdir(input_path)

#iterating over all the fish species directories, converting to .jpg and renaming to a common naming convention
for input_dir in input_dirs:
    img_to_jpg(input_path, input_dir)
    rename_files(input_dir, input_path, input_dir)


#creating our labels df which will be used in one-hot encoding as well as paths to each label for data loading
labels_df = pd.DataFrame(columns=['ID', 'Class'])
label_paths = []

for input_dir in input_dirs:
    dir_items = os.listdir(input_path+input_dir)
    for im in dir_items:
        temp = pd.DataFrame([[im, input_dir]], columns=['ID', 'Class'])
        labels_df = labels_df.append(temp)
        label_paths.append(f'data/{input_dir}/{im}')
        
labels_df = labels_df.reset_index(drop=True)

with open('data_labels/labels', 'wb') as myFile:
    pickle.dump(labels_df, myFile)

with open('data_labels/label_paths', 'wb') as myFile:
    pickle.dump(label_paths, myFile)


#turning every sample into a boolean array and then turning the boolean array into a binary array (or a vector filled with zeros and a single 1), using list comprehension.
species = np.unique(labels_df['Class'])
boolean_labels = [label == species for label in labels_df['Class']]
boolean_labels_int = [label.astype(int) for label in boolean_labels]

with open('data_labels/one_hot_labels', 'wb') as myFile:
    pickle.dump(boolean_labels_int, myFile)


        


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')