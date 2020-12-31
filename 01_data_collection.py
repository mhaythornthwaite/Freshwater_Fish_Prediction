# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:35:50 2020

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#------------------------------ FISH PREDICTION ------------------------------

import time
start=time.time()

from google_images_download import google_images_download   #importing the library


#------------------------------- DATA DOWNLOAD -------------------------------

#15 species of freshwater fish have been chosen in the list below. Initially, 100 images are collected. This will then be manually filtered, removing erranous or misrepresenative images manually.

fish_species = 'Roach fish,Rudd fish,Common Bream,Common Carp,Mirror Carp,Tench,Perch,Pike,Barbel,Chub,Gudgeon,Rainbow Trout,Brown Trout,Grayling'

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":fish_species, "limit":100, "print_urls":False, 'output_directory':'data'}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')
