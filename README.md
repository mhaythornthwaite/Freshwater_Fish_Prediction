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


<!-- ---------------------------------------------------------------------- -->
## Table of Contents

<!--ts-->
* [Aims and Objectives](#Aims-and-Objectives)
* [Data Collection and Preparation](#Data-Collection-and-Preparation)
* [Model Building](#Model-Building)
  - [Multilayer Perceptron](#Multilayer-Perceptron)
  - [Shallow Convolutional Network](#Shallow-Convolutional-Network)
  - [Deep Convolutional Network](#Deep-Convolutional-Network)
  - [Deep Convolutional Network with Data Augmentation](#Deep-Convolutional-Network-with-Data-Augmentation)
  - [Transfer Xception Network](#Transfer-Xception-Network)
* [Further Work and Improvements](#Further-Work-and-Improvements)
<!--te-->


<!-- ---------------------------------------------------------------------- -->
## Aims and Objectives

The objective of this study was to train a network that can identify species of British freshwater fish on RGB images. The image may contain anything (people, animals etc.) providing there is only one species of fish present. No set accuracy was defined for success, rather the study aims to understand what can be achieved with limited data and the best approach to take in this scenario.


<!-- ---------------------------------------------------------------------- -->
## Data Collection and Preparation

In total 14 different species of British freshwater fish were selected, with 100 images for each class collected using the <a href="https://pypi.org/project/google_images_download/" target="_blank"> google_images_download</a> library. Manual QC of this dataset was undertaken to remove any anomalous results returned by the library which were not representative of a given class e.g. images containing multiple different species of fish. The number of samples remaining after this QC per class can be found in figure 1. Raw data and labels can be downloaded <a href="https://drive.google.com/drive/folders/1Sah-IcSeIR8jjLbR2qDgj3RnovsHxEq3?usp=sharing" target="_blank">here</a>.

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures/samples_per_class_barchart.png" alt="Figure 1">

<em>Figure 1. Number of samples per class present in the entire dataset.</em>

<br>

An example of a selection of images in a single batch can be seen in figure 2, standardised with a common resolution of 222 * 222 pixels. Depending on model architecture, in some cases it was necessary to further reduce the resolution of the images in a bid to reduce model complexity and number of trainable parameters.   

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures/Images_in_a_single_batch_v2.png" alt="Figure 2">

<em>Figure 2. Example of images present in a single 32 image batch for training.</em>


<!-- ---------------------------------------------------------------------- -->
## Model Building

In order to achieve the project aims, numerous models of increasing complexity were trained, starting with a basic MLP before training CNN with data augmentation and transfer learning.


<!-- ---------------------------------------------------------------------- -->
### Multilayer Perceptron

To begin with the most basic approach was taken, training a simple multilayer perceptron. Images were resampled to 32 * 32 with a single black-white channel, before being vectorised ready for input into the model. A small image size was chosen in an attempt to reduce the number of trainable parameters. Identifying fish species with the human eye at this resolution is difficult, but not impossible, in most cases a reasonable guess could be made. Model architecture, including the vectorisation stage can be seen in figure 3. 

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures/MLP_Architecture.png" alt="Figure 3">

<em>Figure 3. Simple MLP architecture formed of 4 fully connected dense layers, all with a relu activation function except the output layer which has a softmax activation. </em> 

Training was completed in batch sizes of 32, using <a href="https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class" target="_blank"> categorical crossentropy</a> and an <a href="https://keras.io/api/optimizers/adam/" target="_blank"> adam</a> optimiser. The results can be studied in figure 4. Accuracy peaks at around 8% which as expected is very poor, only narrowly beating a random guess.

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/3_shallow_nn.png" alt="Figure 4">

<em>Figure 4. Accuracy and loss plots, with results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation. Validation accuracy suggests the model performs only marginally better than a random guess. Overfitting of the loss function on the validation data appears after around 40 epochs. </em>


<!-- ---------------------------------------------------------------------- -->
### Shallow Convolutional Network

A simple and relatively shallow convnet was tested next, the architecture of which can be studied in figure 5.  

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures/CNN_Architecture_Shallow.png" alt="Figure 5">

<em>Figure 5. Three convolutional layers and two max pooling layers form the basis of the convolutional block. This is then flattened and connected to 2 dense layers for classification. All convolutional filters are 3 * 3 with a relu activation function. </em>

To reduce the overfitting in this model, L2 regularisation was added to the loss function, and a modest dropout layer (rate = 0.2) added to the model after flattening. This resulted in a decrease in overfitting seen in the validation loss, whilst retaining the same validation accuracy when compared to with models trained without regularisation. Images were also resampled to higher resolutions for training however, the original 32 * 32 resulted in the highest accuracy. This is likely because the network is too shallow 'see' enough of the image when resolution is increased. With this configuration we see accuracy peak at 22% as seen in figure 6.

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/4a_shallow_cnn.png" alt="Figure 6">

<em>Figure 6. Accuracy and loss plots, with results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation. Validation accuracy rises to 22% after 90 epochs. Overfitting of the loss function on the validation data appears after around 30 epochs. </em>


<!-- ---------------------------------------------------------------------- -->
### Deep Convolutional Network

Given the conclusions of the previous model build, a deeper network was tested next, the architecture of which can be studied in figure 7.

<br>
<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures/CNN_Architecture_Deep.png" alt="Figure 7">

<em>Figure 7. Seven convolutional layers and three max pooling layers form the basis of the convolutional block. This is then flattened using a global average pooling layer  and connected to 2 dense layers for classification. All convolutional filters are 3 * 3 with a relu activation function. </em>

Increasing the depth of the model allowed the image size to be increased to 222 * 222 for training. This results in a significant increase in model performance, with accuracy approximately doubling to 42%.  

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/4b_deep_cnn.png" alt="Figure 8">

<em>Figure 8. Accuracy and loss plots, with results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation. Validation accuracy rises to 42% after 100 epochs. Overfitting of the loss function on the validation data appears after around 50 epochs. </em>


<!-- ---------------------------------------------------------------------- -->
### Deep Convolutional Network with Data Augmentation

The next step taken to improve model performance was through introducing data augmentation to the training dataset. When a batch is being 'assembled' ready for training, each image has been allowed to rotate, shift position, zoom, shear and or flip. In each epoch the model is exposed to a different version of every image used in training, helping the model to generalise better and prevent overfitting. Therefore, the model will be better equipped to identify fish species irrespective of the relative size or rotation of the target. An example of data augmentation on a single training image can be seen in figure 9. 

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//data_aug_example.png" alt="Figure 9">

<em>Figure 9. Data augmentation on a single image at the resolution used for training (222 * 222). </em>

The architecture of the model has remained the same as the previous. The introduction of data augmentation alone produces a sizeable increase in accuracy to 55%, seen in figure 10. 

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/5_deep_cnn_aug.png" alt="Figure 10">

<em>Figure 10. Accuracy and loss plots, with results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation. Validation accuracy rises to 55% after 240 epochs. Overfitting of the loss function on the validation data appears after around 170 epochs. </em>


<!-- ---------------------------------------------------------------------- -->
### Transfer Xception Network

Next, an alternative approach is taken using transfer learning. In this architecture, a pre-trained convolutional base has been connected to a flatten, dropout and two trainable dense layers for classification. Various different pretrained networks <a href="https://keras.io/api/applications/" target="_blank">available through keras</a> were tested as the base, with xception chosen due to high model performance. As before, various input resolutions were tested, with the default image size for this base (299 * 299) selected. Using this configuration, we see a significant increase in validation accuracy compared to all previous models presented in this study, a rise to 72% as seen in figure 8. Much higher image resolutions are possible with this architecture because there are only two trainable dense layers. The untrainable convolutional base also likely generalises well to the problem of fish identification due to similarities in the <a href="https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/" target="_blank">classes</a> of the imagenet database which it was trained on (already contains several classes of saltwater fish).

<img src="https://raw.githubusercontent.com/mhaythornthwaite/Freshwater_Fish_Prediction/master/figures//combined_figures_for_report/6_xception.png" alt="Figure 11">

<em>Figure 11. Accuracy and loss plots, with results averaged from 10 models trained with random weights initialisation. Grey shade shows one standard deviation. Validation accuracy rises to 72% after 100 epochs. We see no inflection in the validation loss, suggesting we have little or no overfitting. </em>


<!-- ---------------------------------------------------------------------- -->
## Further Work and Improvements


<!-- ---------------------------------------------------------------------- -->
## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 




<!--
![image info](./figures/samples_per_class_barchart.PNG)
-->