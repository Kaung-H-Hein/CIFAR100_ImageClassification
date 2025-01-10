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

For this project, 15 classes were selected to build and train the models. 

## Outcomes
- Demonstrates a variety of computer vision techniques, from traditional to modern deep learning approaches.
- Highlights the progression from hand-crafted feature methods to advanced neural network-based solutions.
- Provides a clear comparison of the strengths and limitations of each approach on the same dataset.

## 1. Hand-Crafted Features Approach
### Hand-Crafted Features Approach  
This section details the hand-crafted features-based approach for image classification, which involved extensive preprocessing, feature extraction, and model development. Below is a step-by-step description:  

### Data Preprocessing  
The data preprocessing pipeline was essential for preparing the CIFAR-100 dataset for feature extraction and classification:  
  1. **Initial Data Inspection**: Conducted an overview of the dataset to understand its structure and distribution.  
  2. **Class Selection**: Randomly selected 15 classes from the CIFAR-100 dataset for the study.  
  3. **Image Visualisation**: Visualised sample images from the selected classes to ensure proper selection.  
  4. **Image Resizing**: Resized images to 160×160 pixels using `cv2` to standardise input dimensions.  
  5. **Colour Space Conversion**: Converted images to grayscale for feature extraction, focusing on structural details.  
  6. **Contrast Enhancement**: Applied techniques to enhance image contrast, improving feature detection accuracy.  

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
To evaluate the performance of these models, confusion matrices were generated and analysed. 

### Key Libraries Used  
- **PyTorch and Torchvision**: For data handling and preprocessing.  
- **OpenCV (cv2)**: For image resizing and preprocessing.  
- **Scikit-Image**: For feature extraction (SIFT, ORB, Fisher vectors).  
- **Scikit-learn**: For clustering (KMeans), model training (LinearSVC), and evaluation (confusion matrices).  

This approach showcases the potential of traditional computer vision methods for image classification, providing a strong baseline for comparison with neural network-based techniques.
