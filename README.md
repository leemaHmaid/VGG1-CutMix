# VGG11 Implementation from Scratch with CutMix Data Augmentation

## Overview

This repository contains an implementation of the VGG11 architecture from scratch, trained on the ImageNet dataset. The project utilizes the CutMix data augmentation technique alongside the basic data augmentation approaches used in the original VGG11 paper. The main goal is to evaluate the advantages of the CutMix approach.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data Augmentation](#data-augmentation)
  - [Basic Transformations](#basic-transformations)
  - [CutMix Implementation](#cutmix-implementation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributors](#contributors)
- [License](#license)

## Introduction

The VGG11 architecture is a convolutional neural network model that achieves excellent performance on image recognition tasks. This project extends the traditional training process by incorporating CutMix, a regularization technique that improves model robustness by mixing parts of different images and their labels during training.

## Project Structure

The repository consists of the following files:

- `dataset.py`: Handles the loading and preprocessing of the ImageNet dataset.
- `cutmix.py`: Implements the CutMix data augmentation technique.
- `model.py`: Defines the VGG11 model architecture.
- `validate.py`: Contains the validation function to evaluate the model's performance.
- `train.py`: Implements the training loop using SGD optimizer and learning rate scheduler.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vgg11-cutmix.git
   cd vgg11-cutmix
