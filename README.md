# CIFAR100-ImageClassification
This repository contains an image classification project using the CIFAR-100 dataset to explore and compare three image classification approaches:
1. Hand-Crafted Features
2. Neural Networks (NN)
3. Convolutional Neural Networks (CNN)

This project was part of my MSc in Artificial Intelligence and demonstrates skills in computer vision, model development, and performance evaluation.

## Dataset Overview
The CIFAR-100 dataset consists of:
- 100 Classes: Each with 600 images (500 training, 100 testing).
- 20 Superclasses: Classes grouped into broader categories.
- Fine Labels: Specific class labels.
- Coarse Labels: Grouped superclass labels.

For this project, 15 classes were randomly selected to build and train the models. 

## Project Outcomes
- Showcases a variety of computer vision techniques, from traditional hand-crafted features to modern deep learning approaches.
- Highlights the progression from feature extraction methods to data-driven neural networks.
- Provides a clear comparison of the strengths, limitations, and performance of each approach on the CIFAR-100 dataset.
- Demonstrates how different methods address challenges like data variability, computational complexity, and model generalisation.

## 1. Hand-Crafted Features Approach
### Data Preprocessing  
The preprocessing pipeline included the following steps: 
  1. **Initial Data Inspection**: Conducted an overview of the dataset to understand its structure and distribution.  
  2. **Class Selection**: Randomly selected 15 classes from the CIFAR-100 dataset.  
  3. **Image Resizing**: Resized images to 160×160 pixels using `cv2` to standardise input dimensions.  
  4. **Colour Space Conversion**: Converted images to grayscale for feature extraction, focusing on structural details.  
  5. **Contrast Enhancement**: Applied techniques to enhance image contrast, improving feature detection accuracy.  

### Model Development  
Three models were built using different hand-crafted feature extraction methods:  

1. **Bag of Words (BoW) Using SIFT**:  
   - **Feature Extraction**: Used the Scale-Invariant Feature Transform (SIFT) algorithm to detect and describe local keypoints.  
   - **Visual Dictionary**: Built a visual dictionary using KMeans clustering on extracted features.  
   - **Feature Representation**: Encoded images into BoW representations based on the visual dictionary.  
   - **Classification**: Trained a `LinearSVC` classifier on the BoW features for image classification.  

2. **Bag of Words (BoW) Using ORB**:  
   - Similar to the SIFT-based pipeline but employed ORB (Oriented FAST and Rotated BRIEF) for feature detection and description.  

3. **Fisher Vector Representation**:  
   - **Feature Extraction**: Extracted descriptors using SIFT.  
   - **Fisher Vector Encoding**: Applied Fisher vector representation using a Gaussian Mixture Model (GMM) with K-modes.  
   - **Classification**: Trained a `LinearSVC` on Fisher vector-encoded features.  

### Evaluation  
To evaluate the performance of these models, confusion matrices were generated and analysed. This approach showcases the potential of traditional computer vision methods for image classification, providing a strong baseline for comparison with neural network-based techniques.

### Key Libraries Used  
- **PyTorch and Torchvision**: For data handling and preprocessing.  
- **OpenCV (cv2)**: For image resizing and preprocessing.  
- **Scikit-Image**: For feature extraction (SIFT, ORB, Fisher vectors).  
- **Scikit-learn**: For clustering (KMeans), model training (LinearSVC), and evaluation (confusion matrices).

## 2. Neural Network (NN) Approach
### Data Preprocessing
The preprocessing pipeline included the following steps:
  1. **Class Selection**: Randomly selected 15 classes from the CIFAR-100 dataset.
  2. **Image Resizing**: Resized images to 160×160 pixels using `cv2` to standardise input dimensions.
  3. **Colour Space Conversion**: Converted images to grayscale to reduce complexity.
  4. **Contrast Enhancement**: Improved image contrast for better feature representation.
  5. **Tensor Transformation**: Transformed images into tensors for PyTorch compatibility.
  6. **Normalization**: Scaled pixel values to a range of (0-1) suitable for the model.
  7. **Dataset Preparation**: Prepared training and testing datasets, remapped class indices, and batched data for efficient processing.

### Model Architecture
A simple custom neural network was designed with the following structure:
  - **Flatten Layer**: Converts 2D images into 1D vectors.
  - **Hidden Layer**: Fully connected layer with 500 hidden units.
  - **Output Layer**: Fully connected layer mapping to the 15 selected classes.

### Training and Evaluation
**Training Setup**:
- **Loss Function**: CrossEntropyLoss for multi-class classification.
- **Optimiser**: Stochastic Gradient Descent (SGD) with a learning rate of 0.0001.
- **Epochs**: 20, with progress tracked using tqdm.
  
**Training Procedure**:
- Forward pass, loss calculation, backpropagation, and parameter updates were performed iteratively.
- Average loss per epoch was calculated for performance monitoring.

### Evaluation:
The model was evaluated on the test dataset using a custom evaluation function.

**Metrics**:
- **Loss**: Average classification loss.
- **Accuracy**: Proportion of correctly predicted labels.

Results were summarised, and a confusion matrix was generated for error analysis.

This approach effectively showcased the neural network's ability to classify images by learning feature representations with a straightforward architecture.

## 3. Convolutional Neural Network (CNN) Approach

In the model architecture follows the **TinyVGG** design, which consists of two convolutional blocks and a fully connected classifier, as described in the original [CNN Explainer](https://poloclub.github.io/cnn-explainer/).

### Data Preprocessing
  1. **Class Selection**: Randomly selected 15 classes from the CIFAR-100 dataset.
  2. **Image Resizing**: Resized images to 160×160 pixels using `cv2` to standardise input dimensions.
  3. **Grayscale Conversion**: The images were converted to grayscale to simplify the model and reduce computational load.
  4. **Contrast Enhancement**: Contrast of the images was enhanced for better feature extraction.
  5. **Tensor Transformation**: The images were transformed into tensors to be processed by PyTorch.
  6. **Normalization**: The pixel values were normalised to the range of 0 to 1 to improve model convergence.
  7. **Dataset Preparation**: The dataset was prepared by remapping the indexes of the selected classes, batching the data for training.

### Model Architecture
The model, **CIFAR100_V2**, consists of the following key components:
1. **Convolutional Blocks**:  
   - **Block 1**: Contains two convolutional layers followed by ReLU activations and max-pooling.
   - **Block 2**: Contains another two convolutional layers with ReLU activations and max-pooling.
2. **Fully Connected Classifier**:  
   - A flattened layer followed by a fully connected layer that outputs the final class predictions.

### Training and Evaluation
- **Loss Function**: Cross-entropy loss was used as the objective function to optimise the model.
- **Optimizer**: Stochastic Gradient Descent (SGD) was used with a learning rate of 0.001 to update model parameters.
- **Accuracy**: The accuracy of the model was tracked during both training and testing phases.
- **Training**: The model was trained over 16 epochs, with the loss and accuracy being calculated at each epoch.
- **Testing**: After each epoch, the model was evaluated on the test set to measure its performance on unseen data.

The CNN approach allows the model to automatically learn spatial hierarchies in images, making it well-suited for image classification tasks. The use of multiple convolutional layers helps in capturing complex features such as edges, textures, and patterns in the CIFAR-100 images.
