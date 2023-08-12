# Smart-Attendance
Smart Attendance using Deep Learning &amp; Computer Vision

![Python](https://img.shields.io/badge/-Python-black?style=flat&logo=python)
![Deep Learning](https://img.shields.io/badge/-Deep%20Learning-566be8?style=flat)
![Tensorflow](https://img.shields.io/badge/-Tensorflow-gray?style=flat&logo=tensorflow)
![Keras](https://img.shields.io/badge/-Keras-gray?style=flat&logo=keras)
![OpenCV](https://img.shields.io/badge/-OpenCV-gray?style=flat&logo=opencv)
![NumPy](https://img.shields.io/badge/-NumPy-gray?style=flat&logo=numpy)
![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-black?style=flat&logo=jupyter)
![Streamlit](https://img.shields.io/badge/-Streamlit-f0806c?style=flat)

## Description
SA is a clever way to manage attendance using computers. Instead of saying names or using paper, it takes pictures of people with webcams without having to ask.
It uses webcams to quickly recognize faces in real-time, changing the way attendance is recorded by using Harr Cascade Classifier.
This makes attendance tracking faster and more accurate.It's like having a smart helper that takes care of attendance without any manual effort.

## Steps taken in this project <a name="project-steps"></a>
- Planning
- Data Collection and Preparation
- Evaluation & Validation of model
- User Interface Development for real time usage
  
## Installation requirements 
To run this code, you need to:
- import streamlit as st
- import numpy as np
- import cv2
- import tensorflow as tf
- import os
- from keras.models import model_from_json
- from PIL import Image
- from google.colab import drive
- from datetime import datetime

## Labelling 
| Label | Description |
| --- | --- |
| 0 | Anglina Jolie |
| 1 | Nu Nu Hlaing |
| 2 | Will Smith |

## AI Ethics
* [ ]  **Privacy & Security**:
- Don’t save any input data from users 
- Avoid deploying the system where it could be used for unethical purposes
* [ ]  **Explainability**:
- Easy to use with user-centric design
- People just need to stand in front of a camera - no special cards or lists.
* [ ]  **Fairness**:
- Treat everyone fairly and equally without any bias based on factors like race, gender, or background
* [ ]  **Limitations**:
- Sometimes, it doesn't work as well if someone looks different.
- May vary depending on lighting conditions, camera quality, and camera angle

## Future Work
* [ ]  **Model optimization**:To extend the model to be more culturally sensitive and adaptable
* [ ]  **Dataset expansion**:To expand and diversify the dataset to ensure it represents a wide range of people.
* [ ]  **Data augmentation**:To improve performance in challenging environments, such as low-light conditions or low-resolution images
* [ ]  **Real-time Performance**:To optimize the model for real-time processing to provide immediate updates without delay.

## Refrences
- Emotion detection: https://github.com/juan-csv/emotion_detection
- Face Recognition: https://github.com/juan-csv/face-recognition
- Haar Cascade: https://github.com/opencv/opencv/tree/master/data/haarcascades
- Dataset from Kaggle
- Open Source Libraires
