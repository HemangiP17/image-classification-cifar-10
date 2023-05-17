# CIFAR-10 Image Classification using Convolutional Neural Networks
This project demonstrates the implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images, divided into 10 classes. The goal is to train a CNN model to accurately classify these images into their respective classes.

## Requirements
To run this project, you need the following dependencies:

<ol>
  <li>Numpy</li>
  <li>Matplotlib</li>
  <li>Keras</li>
</ol>

You can install these dependencies using pip:

<pre>
<div class="p-4 overflow-y-auto"><code class="!whitespace-pre hljs">pip install numpy matplotlib keras
</code></div></pre>

## Dataset
The CIFAR-10 dataset is automatically downloaded using the cifar10 module from Keras. It is split into training and testing sets, consisting of 50,000 and 10,000 images, respectively. Additionally, a validation set is created by further splitting the training set.

## Data Preprocessing
Before training the model, the data is preprocessed as follows:
<ul>
<li>The pixel values of the images are normalized to the range of 0 to 1 by dividing them by 255.</li>
<li>The target labels are one-hot encoded using the np_utils.to_categorical function from Keras.</li>
</ul>
  
## Visualization
A function plot_digits is defined to visualize a set of CIFAR-10 images. It takes an array of image instances and displays them in a grid. This function is used to plot a subset of the training images after preprocessing.

## Model Architecture
The CNN model architecture is defined using the Keras Sequential API. The model consists of the following layers:
<ul>
<li>Convolutional layer with 64 filters, kernel size 3x3, and ReLU activation.</li>
<li>MaxPooling layer with pool size 2x2.</li>
<li>Convolutional layer with 128 filters, kernel size 2x2, and ReLU activation.</li>
<li>MaxPooling layer with pool size 2x2.</li>
<li>Flatten layer to convert the 2D feature maps into a 1D vector.</li>
<li>Dropout layer with a dropout rate of 0.5 to prevent overfitting.</li>
<li>Dense (fully connected) layer with 10 units and softmax activation for multiclass classification.</li>
<li>The model is compiled with the mean squared error loss function, the Adam optimizer, and accuracy as the evaluation metric.</li>
</ul>
  
## Model Training
The compiled model is trained using the fit function in Keras. The training data is used for both training and validation, with a specified validation split. The model is trained for 30 epochs.

After training, the history object stores the training and validation loss and accuracy values, which are plotted using Pandas and Matplotlib.

## Model Evaluation
The trained model is evaluated on the validation set using the evaluate function. The evaluation returns the loss value and accuracy of the model on the unseen data.

## Results
The final evaluation of the model on the validation set shows a loss of 0.0360 and an accuracy of 0.7514.

Note: The code provided is written in Python and utilizes the Keras library for deep learning. Make sure to have the required dependencies installed before running the code.
