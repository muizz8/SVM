# SVM (Support Vector Machines) 🤖

## Overview 📝
This repository contains code for implementing Support Vector Machines (SVM) using scikit-learn to classify human cell records as benign or malignant. SVM is a powerful classification algorithm that works by mapping data to a high-dimensional feature space to categorize data points, even when they are not linearly separable. The model finds a separator between categories and transforms the data so that the separator can be represented as a hyperplane. This allows for accurate classification of new data based on its characteristics.

## Objectives 🎯
- Use scikit-learn to implement Support Vector Machines for classification purposes.

## Usage 🚀
To use this code, ensure you have the necessary packages installed. You can install them using the following command:
```
%pip install scikit-learn
```
Then, import the required packages in your Jupyter Notebook:
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
```

## Dataset 📊
The dataset used in this project consists of human cell records and their corresponding classifications as benign or malignant.
