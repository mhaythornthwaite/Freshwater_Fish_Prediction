<h2 align="center"> Freshwater Fish Prediction </h2>


<p align="center">
    <a href="https://www.python.org/doc/" alt="Python 3.8">
        <img src="https://img.shields.io/badge/python-v3.8+-blue.svg" />
    </a>
    <a href="https://github.com/mhaythornthwaite/Freshwater_Fish_Prediction/blob/main/LICENSE.md" alt="Licence">
        <img src="https://img.shields.io/badge/license-MIT-yellow.svg" />
    </a>
    <a href="https://github.com/mhaythornthwaite/Freshwater_Fish_Prediction/commits/main" alt="Commits">
        <img src="https://img.shields.io/github/last-commit/mhaythornthwaite/Freshwater_Fish_Prediction/main" />
    </a>
    <a href="https://github.com/mhaythornthwaite/Freshwater_Fish_Prediction" alt="Activity">
        <img src="https://img.shields.io/badge/contributions-welcome-orange.svg" />
    </a>
</p>



## Table of Contents

<!--ts-->
* [Aims and Objectives](#Aims-and-Objectives)
* [Dataset](#Dataset)
* [Data Cleaning and Preparation](#Data-Cleaning-and-Preparation)
* [Model Building](#Model-Building)
  - [Basic Multilayer Perceptron](#Basic-Multilayer-Perceptron)
  - [Basic Convolutional Network](#Basic-Convolutional-Network)
  - [Complex Convolutional Network](#Deep-Convolutional-Network)
  - [Transfer VG16 Network](#Transfer-VG16-Network)
* [Further Work and Improvements](#Further-Work-and-Improvements)
<!--te-->


## Aims and Objectives

The aim of this study was to train a network that could identify species of british freshwater fish on RGB images. 

## Dataset

In total 14 different species of British freshwater fish were selected, with 100 images for each class collected using the <a href="https://pypi.org/project/google_images_download/" target="_blank"> google_images_download</a> library. Manual QC of this dataset was undertaken to remove any anomalous results returned by the library which were not representative of the given class e.g. images containing multiple different species of fish. The number of samples remaining after this QC per class can be found in figure 1. Raw data and labels can be downloaded <a href="https://drive.google.com/drive/folders/1Sah-IcSeIR8jjLbR2qDgj3RnovsHxEq3?usp=sharing" target="_blank">here</a>.

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures/samples_per_class_barchart.png" alt="Figure 1">

<em>Figure 1. Number of samples per class present in the entire dataset.</em>

## Data Cleaning and Preparation



## Model Building


### Basic Multilayer Perceptron

### Basic Convolutional Network

### Complex Convolutional Network

### Transfer VG16 Network





## Further Work and Improvements



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 




<!--
![image info](./figures/samples_per_class_barchart.PNG)
-->