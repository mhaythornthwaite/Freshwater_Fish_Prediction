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
* [Data Collection & Preparation](#Data-Collection-&-Preparation)
* [Model Building](#Model-Building)
  - [Multilayer Perceptron](#Multilayer-Perceptron)
  - [Convolutional Network](#Convolutional-Network)
  - [Transfer VG16 Network](#Transfer-VG16-Network)
* [Further Work and Improvements](#Further-Work-and-Improvements)
<!--te-->


## Aims and Objectives

The objective of this study was to train a network that could identify species of British freshwater fish on RGB images. The image may contain anything (people, animals etc.) providing there is only one species of fish present. No set accuracy is defined for success, rather the study aims to understand what can be achieved with limited data and the best approach to take in this scenario.


## Data Collection & Preparation

In total 14 different species of British freshwater fish were selected, with 100 images for each class collected using the <a href="https://pypi.org/project/google_images_download/" target="_blank"> google_images_download</a> library. Manual QC of this dataset was undertaken to remove any anomalous results returned by the library which were not representative of a given class e.g. images containing multiple different species of fish. The number of samples remaining after this QC per class can be found in figure 1. Raw data and labels can be downloaded <a href="https://drive.google.com/drive/folders/1Sah-IcSeIR8jjLbR2qDgj3RnovsHxEq3?usp=sharing" target="_blank">here</a>.

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures/samples_per_class_barchart.png" alt="Figure 1">

<em>Figure 1. Number of samples per class present in the entire dataset.</em>


## Model Building

In order to achieve the project aims, numerous models of increasing complexity were trained, starting with a basic MLP before training CNN with data augmentation and transfer learning.


### Multilayer Perceptron

To begin with the most basic approach was taken, training a simple multilayer perceptron. Images were resampled to 32 * 32 before being vectorised ready for input into the model. A small image size was chosen in an attempt to reduce the number of trainable parameters. Identifying fish species with the human eye was difficult, but not impossible, in most cases a reasonable guess could be made. 

The final model comprised of three fully connected hidden layers, all with a relu activation function, beginning with 192 neurons, reducing to 96 and 48 neurons in the later layers. A softmax activation was chosen for the output comprising of 14 neurons. Training was completed in batch sizes of 32, using <a href="https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class" target="_blank"> categorical crossentropy</a> and an <a href="https://keras.io/api/optimizers/adam/" target="_blank"> adam</a> optimiser. 

The results can be studied in figure 2. Overfitting occurs after around 15 epochs where an inflection can be observed in the loss of the validation set. Despite this we see an increase in accuracy to ~ 10% after 60 epochs. Selection of one of these models may produce a better prediction than a random guess but as expected the result is poor.

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/3_basin_nn.png" alt="Figure 2">



<em>Figure 2. (a) Training and validation accuracy up to 100 epochs. Validation accuracy only rises above a random guess accuracy after ~ 60 epochs. (b) Training and validation loss up to 100 epochs. Overfitting of the loss function on the validation data appears after ~ 15 epochs.</em>

### Convolutional Network



### Transfer VG16 Network





## Further Work and Improvements



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 




<!--
![image info](./figures/samples_per_class_barchart.PNG)
-->