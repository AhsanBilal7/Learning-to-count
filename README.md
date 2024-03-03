# ResNet-101 Ellipse Counting Model

This repository contains code for training a ResNet-101 model to count ellipses in images. The model is trained on the MR (mid-resolution) dataset provided in the `shapes_dataset_MR` directory.

## Train-Test Split

The dataset has been split into training and testing sets using a ratio of 80:20. The images have been randomly shuffled and distributed into the respective directories. 

## Naive Bayes Classifier

A Naive Bayes classifier has been implemented to classify the images based on their features. The classifier has been trained on the training set and evaluated on the testing set. The accuracy achieved on the training set is [insert training accuracy], and the accuracy on the testing set is [insert testing accuracy].

## Loading Pre-Trained Model

To load the pre-trained model, the `load_model` function from the `Bayes` class can be used. This function loads the parameters (mean, covariance, and priors) stored in a pickle file.
1. Load gaussian and posr orobabilities by google link [here](https://drive.google.com/drive/folders/19lrEzhurVV_V4UZEmgG3Rab_XS87RO4H?usp=sharing)
    ```bash
    model = Bayes()
    model.load_model('pretrained_model.pkl')


## Prerequisites
- Python 3
- PyTorch
- torchvision
- tqdm

## Pretrained Weights
Pretrained weights for the ResNet-101 model can be downloaded from [Google Drive](https://drive.google.com/drive/folders/19lrEzhurVV_V4UZEmgG3Rab_XS87RO4H?usp=sharing).

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/resnet-101-ellipse-counting.git
2. Install the required packages:


## Dataset Preparation
1. Download the MR dataset from [here](http://george-vogiatzis.org/MScAI/shapes_dataset_MR.zip).
2. Extract the dataset and place it in the `shapes_dataset_MR` directory.
3. Ensure that the dataset is organized in the following structure:
    ```bash
    shapes_dataset_MR
    ├── labels.csv
    └── train
    ├── class_1
    │ ├── image1.png
    │ ├── image2.png
    │ └── ...
    ├── class_2
    │ ├── image1.png
    │ ├── image2.png
    │ └── ...
    └── ...
## Training
To train the ResNet-101 model, run the following command:


The trained model weights will be saved as `resnet_101_MR.pth`.

## Testing
To test the trained model on a test set, run the following command:




## Evaluation
You can evaluate the model performance by computing accuracy metrics using the test results.

## Acknowledgments
- CS231n Convolutional Neural Networks for Visual Recognition
- Deep Learning with PyTorch: A 60 Minute Blitz
- PyTorch
- Goodfellow, Bengio, and Courville. Deep Learning. MIT Press, 2016.
- ImageNet Large Scale Visual Recognition Challenge
