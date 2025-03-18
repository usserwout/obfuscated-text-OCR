# Obfuscated-text-OCR

## Overview

This project is designed to detect and read distorted text from images, specifically for the purpose of identifying and extracting coupon codes. The system uses  deep learning techniques, including Convolutional Neural Networks (CNNs) and Residual Networks (ResNet), to accurately identify and classify characters in challenging visual conditions.

## Key Features

- **Character Detection and Classification**: Uses Mask R-CNN with a ResNet backbone to detect and classify characters in images.
- **Custom Dataset Handling**: Includes a custom dataset class for loading images and annotations, and preprocessing them for training.
- **Image Transformation**: Applies various transformations to images to prepare them for model input.
- **Training and Inference**: Provides scripts for training the model on a dataset and for running inference to detect characters in new images.
- **Visualization**: Includes functionality to visualize transformed images and model predictions for debugging and analysis.

## Techniques Used

- **Mask R-CNN**: A state-of-the-art object detection model that extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI).
- **ResNet (Residual Network)**: A deep neural network architecture that uses residual connections to enable the training of very deep networks. This project uses ResNet as the backbone for the Mask R-CNN model.
- **Convolutional Neural Networks (CNNs)**: Used for feature extraction from images, enabling the model to learn complex patterns and representations.
- **Data Augmentation and Preprocessing**: Includes various techniques for augmenting and preprocessing images to improve model robustness and performance.

## Example Process

Here is an example of the process an image goes through in this project:

**Input Image**:

   ![Input Image](example/input.png)

1. **Text position detection**:
   ![text_detection](example/text_detection.png)

2. **Denoising**:
   - Generate a mask:
     ![Preprocessing Mask](example/mask.png)
   - Apply mask to image:
     ![Denoised Image](example/denoised.png)

3. **Output Image**:
   ![Output Image](example/output.png)

With following predictions
```
 1. 8 (conf: 0.72) at [9, 9] -> [22, 39]
 2. V (conf: 0.75) at [22, 9] -> [38, 39]
 3. 8 (conf: 0.85) at [38, 8] -> [55, 38]
 4. S (conf: 0.90) at [52, 8] -> [74, 38]
 5. Q (conf: 0.96) at [95, 9] -> [113, 42]
 6. V (conf: 0.84) at [112, 9] -> [129, 39]
 7. M (conf: 0.87) at [127, 8] -> [153, 38]
 8. X (conf: 0.94) at [153, 8] -> [171, 38]
 9. M (conf: 0.99) at [196, 8] -> [219, 38]
10. S (conf: 0.77) at [219, 8] -> [236, 37]
11. F (conf: 0.99) at [236, 8] -> [250, 37]
12. W (conf: 0.99) at [249, 8] -> [273, 40]
```   

## Acknowledgements

This project leverages several powerful libraries and frameworks, including:

- **PyTorch**: For building and training the deep learning models.
- **Torchvision**: For pre-trained models and image transformations.
