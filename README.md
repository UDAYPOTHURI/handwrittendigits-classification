# Handwritten Digit Classification using CNN

This project aims to classify images of handwritten digits (MNIST dataset) using a custom Convolutional Neural Network (CNN) built with PyTorch. The project demonstrates the steps of data preprocessing, model building, training, and evaluation.

## Project Structure

- `handwritten_digit_classification.ipynb`: Jupyter notebook containing the complete workflow from data loading to model training and evaluation.
- `data/`: Directory containing the dataset.
- `README.md`: Documentation for the project.

## Dataset

The dataset used is the MNIST dataset, which contains images of handwritten digits (0-9), divided into training and testing sets.

- `data/train/`: Training images
- `data/test/`: Testing images

## Preprocessing

The preprocessing steps include:

1. Loading the dataset from the specified directories.
2. Applying transformations such as random horizontal flip, color jitter, random rotation, and conversion to tensor.

## Model

### Model Architectures

#### `handwrittenv0`

A simple CNN with:

- Two convolutional layers followed by ReLU activations.
- Two max-pooling layers.
- A dropout layer.
- A flattening layer and a linear layer.

#### `handwrittenv1`

An enhanced CNN with:

- Three convolutional layers with ReLU activations.
- Two max-pooling layers.
- A dropout layer.
- A flattening layer and a linear layer.

### Training

The training process involves:

1. Defining the loss function (CrossEntropyLoss) and optimizer (SGD).
2. Training the model for a specified number of epochs (3 in this case).
3. Recording the training and testing losses and accuracies for each epoch.

### Evaluation

The model's performance is evaluated by printing the training and testing accuracies and losses over the epochs.

## Dependencies

- torch
- torchvision
- matplotlib
- torchinfo
- Pillow (PIL)

## Results

The training and testing losses and accuracies are recorded and can be visualized in the notebook.

## Conclusion

This project demonstrates how to build and train custom CNNs to classify images of handwritten digits. Further improvements can be made by tuning the model architecture, experimenting with different loss functions and optimizers, and using more advanced data augmentation techniques.
