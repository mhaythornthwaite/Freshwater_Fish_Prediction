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

<br>

An example of a selection of images in a single batch can be seen in figure 2, standardised with a common resolution of 256 * 256 pixels. Depending on model architecture, in some cases it was necessary to further reduce the resolution of the images in a bid to reduce model complexity and number of trainable parameters.   

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures/Images_in_a_single_batch_v2.png" alt="Figure 2">

<em>Figure 2. Example of images present in a single batch for training.</em>


## Model Building

In order to achieve the project aims, numerous models of increasing complexity were trained, starting with a basic MLP before training CNN with data augmentation and transfer learning.


### Multilayer Perceptron

To begin with the most basic approach was taken, training a simple multilayer perceptron. Images were resampled to 32 * 32 with a single black-white channel, before being vectorised ready for input into the model. A small image size was chosen in an attempt to reduce the number of trainable parameters. Identifying fish species with the human eye was difficult, but not impossible, in most cases a reasonable guess could be made. 

The final model comprised of three fully connected hidden layers, all with a relu activation function except the output layer which has a softmax activation. Training was completed in batch sizes of 32, using <a href="https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class" target="_blank"> categorical crossentropy</a> and an <a href="https://keras.io/api/optimizers/adam/" target="_blank"> adam</a> optimiser. 

The results can be studied in figure 3. Overfitting occurs after around 40 epochs where an inflection can be observed in the loss of the validation set. Despite this we see the largest accuracy of 8% after 80 epochs. Selecting a model after 80 epochs may produce the best absolute prediction, but due to overfitting seen in the loss function, will likely overestimate the probability of the prediction. Depending on the use of the model, this may or may not pose an issue. As expected however, model performance is very poor, only narrowly beating a random guess.

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/3_basin_nn_v2.png" alt="Figure 3">

<em>Figure 3. (a) Results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation around the validation accuracy. Accuracy suggests the model performs only marginally better than a random guess. (b) Grey shade shows one standard deviation around the validation loss. Overfitting of the loss function on the validation data appears after 40 epochs. </em>


### Convolutional Network

A simple convnet was tested next. 3 convolutional layers and 2 max pooling layers form the basis of the convolutional block. This is then flattened and connected to 2 dense layers for classification.

Images were resampled to various sizes for training however, the original 32 * 32 resolution resulted in highest accuracy and least overfitting. As seen in figure 4, using a convolutional architecture results in a doubling of accuracy to 19%. 

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/4a_basic_convnet_v2.png" alt="Figure 4">

<em>Figure 4. (a) Results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation around the validation accuracy. Validation accuracy rises to 19% after 90 epochs. (b) Grey shade shows one standard deviation around the validation loss. Overfitting of the loss function on the validation data appears after around 20 epochs. Note that the validation loss for the first epoch is lower than the training loss. This is because training loss is measured during each epoch whereas validation loss is measured after. </em>

<br>

In an attempt to reduce the overfitting in this model, L2 regularisation was added to the loss function, and a modest dropout layer (rate = 0.2) added to the model after flattening. Accuracy remains roughly the same, increasing slightly to 20%. We do however, see a decrease in overfitting apparent in the validation loss, seen in figure 5.

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/4b_basic_convnet_reg_v2.png" alt="Figure 5">

<em>Figure 5. (a) Results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation around the validation accuracy. Validation accuracy rises to 20% after 90 epochs. (b) Grey shade shows one standard deviation around the validation loss. Overfitting of the loss function on the validation data appears after around 30 epochs and increases to a lower value. As with the previous model the validation loss for the first epoch is lower than the training loss. The addition of dropout could also be contributing to this effect, as this layer is only active during the training phase. </em>

<br>

The last step to improve this model was through introducing data augmentation to the training dataset. When a batch is being 'assembled' ready for training, each image is allowed to rotate, shift position, zoom, shear and or flip. In each epoch the model is exposed to a different version of every image used in training, helping the model to generalise better and prevent overfitting. As a result, the model will be better equipped to identify fish species irrespective of the relative size or rotation of the target. An example of data augmentation on a single training image can be seen in figure 6. 

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//data_aug_combined.png" alt="Figure 6">

<em>Figure 6. (a) Data augmentation on a single high-resolution image (512 * 512) (b) Data augmentation on the same image but with a resolution of 64 * 64 used for training. </em>

<br>

The architecture of the model remains the same as the previous. As before, input image sizes were tested as an important hyperparameter, with 64 * 64 yielding highest accuracy. This is a trade-off between retaining as much information stored in the original image whilst keeping the number of trainable parameters in the model to a minimum. This stage resulted in a good increase in accuracy to 42% as shown in figure 7. 

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/5_basic_convnet_aug_v2.png" alt="Figure 7">

<em>Figure 7. (a) Results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation around the validation accuracy. Validation accuracy rises to 42% after 250 epochs. (b) Grey shade shows one standard deviation around the validation loss. Overfitting of the loss function on the validation data appears after around 75 epochs. </em>


### Transfer VG16 Network





## Further Work and Improvements



## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 




<!--
![image info](./figures/samples_per_class_barchart.PNG)
-->