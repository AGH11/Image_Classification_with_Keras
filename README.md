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
