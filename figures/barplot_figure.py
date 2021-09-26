# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 16:46:45 2021

@author: mhayt
"""


print('\n\n ---------------- START ---------------- \n')

#------------------------------ FISH PREDICTION -------------------------------

import time
start=time.time()

import os
import matplotlib.pyplot as plt

plt.close('all')


#-------------------------------- QUICK FIGURE --------------------------------


fish_species = ['Roach', 'Rudd', 'Common_Bream', 'Common_Carp', 'Mirror_Carp', 'Tench', 'Perch', 'Pike', 'Barbel', 'Chub', 'Gudgeon', 'Rainbow_Trout', 'Brown_Trout', 'Grayling']

fish_species_v2 = ['Roach', 'Rudd', 'Common Bream', 'Common Carp', 'Mirror Carp', 'Tench', 'Perch', 'Pike', 'Barbel', 'Chub', 'Gudgeon', 'Rainbow Trout', 'Brown Trout', 'Grayling']

samples_in_class = []

directory_path = '../data'
for fish in fish_species:
    no_of_files = len(os.listdir(directory_path+'/'+fish))
    samples_in_class.append(no_of_files)
    

#fig setup including rotation of xticks
fig, ax = plt.subplots(figsize=(12, 5))
#fig.suptitle('Samples Per Class', y=0.95, fontsize=14, fontweight='bold')
plt.setp(ax.xaxis.get_majorticklabels(), rotation=55)
plt.subplots_adjust(bottom=0.23)

#plotting training and validation loss
for i, sample in enumerate(samples_in_class):
    ax.bar(fish_species_v2[i], sample, color='lightcoral', edgecolor='rosybrown')
    


# ----------------------------------- END -------------------------------------

print('\n', 'Script runtime:', round(((time.time()-start)/60), 2), 'minutes')
print(' ----------------- END ----------------- \n')