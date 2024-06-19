# VGG11 Implementation from Scratch with CutMix Data Augmentation

## Overview

This repository contains an implementation of the VGG11 architecture from scratch, trained on the ImageNet dataset. The project utilizes the CutMix data augmentation technique alongside the basic data augmentation approaches used in the original VGG11 paper. The main goal is to evaluate the advantages of the CutMix approach.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Data Augmentation](#data-augmentation)
  - [Basic Transformations](#basic-transformations)
  - [CutMix Implementation](#cutmix-implementation)
 
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

## CutMix Implementation

### Overview
CutMix is an advanced data augmentation technique that goes beyond traditional methods by mixing parts of multiple images and their corresponding labels. This method encourages the model to learn from diverse image regions and labels, thereby enhancing its ability to generalize to unseen data.

### How CutMix Works
- Random Selection: Two images are randomly selected from the dataset.
- Random Bounding Box: A random bounding box is selected from one of the images.
- Patch Mixing: The selected bounding box in the first image is replaced with a patch from the second image.
- Label Mixing: The labels of the original and patch images are mixed proportionally to the area of overlap between the bounding box and the patch.
- CutMix helps in regularizing the model, reducing overfitting, and improving its robustness to variations in the dataset.

