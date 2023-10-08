# Convolutional Neural Network (CNN) for CIFAR-10 Classification

This repository contains a simple Convolutional Neural Network (CNN) implemented using Keras for classifying images from the CIFAR-10 dataset.

## Dependencies
- Keras
- NumPy
- TensorFlow

## Usage
1. Install the required dependencies:

   ```bash
   pip install keras numpy tensorflow
   ```

2. Run the provided Python script to train the model:

   ```python
   python train_model.py
   ```

## Model Architecture
The CNN architecture consists of convolutional layers, max-pooling layers, dropout layers, and dense layers. The model is summarized using `model.summary()`.

## Training
The model is trained on the CIFAR-10 dataset for 30 epochs with a batch size of 64. The training process includes data preprocessing and augmentation.

## Model Saving
The trained model is saved in two parts:
- `model_structure.json`: JSON file containing the model architecture.
- `model_weights.h5`: HDF5 file containing the trained model weights.

Feel free to modify the script and experiment with different hyperparameters for further improvement.



# CIFAR-10 Image Classification - Inference

This repository provides a simple script for classifying images using a pre-trained Convolutional Neural Network (CNN) on the CIFAR-10 dataset.

## Usage
1. Ensure you have the required dependencies installed:

   ```bash
   pip install keras numpy
   ```

2. Download the pre-trained model files (`model_structure.json` and `model_weights.h5`).

3. Run the provided Python script to classify an image:

   ```python
   python classify_image.py
   ```

## Model Loading and Inference
The script loads the pre-trained model architecture from `model_structure.json` and weights from `model_weights.h5`. It then uses the model to classify a sample image (`frog.png` by default).

The predicted class label and likelihood are printed to the console.

Feel free to replace the sample image with your own and explore the model's predictions.

**Class Labels:**
- Plane
- Car
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Boat
- Truck
