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

#----------------------------- DATA REFORMATTING -----------------------------

input_path = 'C:/Users/mhayt/Documents/Software_Developer-Python/Freshwater_Fish_Prediction/data/'
input_dirs = os.listdir(input_path)

#iterating over all the fish species directories, converting to .jpg and renaming to a common naming convention
for input_dir in input_dirs:
    img_to_jpg(input_path, input_dir)
    rename_files(input_dir, input_path, input_dir)



# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')