# VGG11 Implementation from Scratch with CutMix Data Augmentation

## Overview

This repository contains an implementation of the VGG11 architecture from scratch, trained on the ImageNet dataset. The project utilizes the CutMix data augmentation technique alongside the basic data augmentation approaches used in the original VGG11 paper. The main goal is to evaluate the advantages of the CutMix approach.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data Augmentation](#data-augmentation)
  - [Basic Transformations](#basic-transformations)
  - [CutMix Implementation](#cutmix-implementation)
  - [Overview](#Overview)
  - [How CutMix Works](#How-CutMix-Works)
- [Dataset](#Dataset)
- [Model Configuratio](#Model-Configuratio)
- [Conclusion](#Conclusion)
 
## Introduction

The VGG11 architecture is a convolutional neural network model that achieves excellent performance on image recognition tasks. This project extends the traditional training process by incorporating CutMix, a regularization technique that improves model robustness by mixing parts of different images and their labels during training.

## Project Structure

The repository consists of the following files:

- `CustomImageDataset.py`: Handles the loading and preprocessing of the ImageNet dataset.
- `cutmix.py`: Implements the CutMix data augmentation technique.
- `model.py`: Defines the VGG11 model architecture.
- `validate.py`: Contains the validation function to evaluate the model's performance.
- `train.py`: Implements the training loop using SGD optimizer and learning rate scheduler.
  
## Data Augmentation
## Basic Transformations:
- Random Cropping.
- Color jittering.
- Image resize

## CutMix Implementation
 ![cutmixDA](https://github.com/leemaHmaid/VGG1-CutMix/assets/52715254/c010f5f9-6ba0-46c7-8837-95634aaac367)

## Overview
CutMix is an advanced data augmentation technique that goes beyond traditional methods by mixing parts of multiple images and their corresponding labels. This method encourages the model to learn from diverse image regions and labels, thereby enhancing its ability to generalize to unseen data.

## How CutMix Works
- Random Selection: Two images are randomly selected from the dataset.
- Random Bounding Box: A random bounding box is selected from one of the images.
- Patch Mixing: The selected bounding box in the first image is replaced with a patch from the second image.
- Label Mixing: The labels of the original and patch images are mixed proportionally to the area of overlap between the bounding box and the patch.
- CutMix helps in regularizing the model, reducing overfitting, and improving its robustness to variations in the dataset.
## Dataset

ImageNet is a vast dataset containing over 1.2 million annotated images distributed across
1000 object classes. It is commonly used to train and evaluate computer vision models due
to its size and diversity.

##Model Configuratio:
• Convolutional Layers: 8 layers with 3x3 filters.
• ReLU Activation: After each convolutional layer.
• MaxPooling Layers:After certain convolutional layers for dimensionality reduction.
• Fully Connected Layers:: 3 layers with 4096, 4096, and num_classes units
respectively, with ReLU and Dropout

## Conclusion
In this project, we implemented the VGG11 architecture from scratch and incorporated
both basic data augmentation techniques and the CutMix data augmentation method.The
aim of this project is demonstrate the difference after using cutmix as data augmentation
and to show the improvements vs. basic data augmentation.Due to hardware limitations,
specifically limited GPU resources, we were only able to train the VGG11 model on the
ImageNet dataset for 5 epochs, despite the intention to train for 75 epochs. The results
reflect the initial phase of the training process and provide insights into the model’s learning
behavior under constrained conditions.

