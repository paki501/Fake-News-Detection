# Fake-News-Detection
This project aims to classify news articles as Fake or Real using Natural Language Processing (NLP) techniques and Machine Learning/Deep Learning models. We use Random Forest and LSTM for classification, leveraging Python's scikit-learn and TensorFlow libraries.

# Getting Started
Follow these instructions to set up and run the project on your local machine for development and testing.
## Prerequisites
To run this project, you need to install the following:

+ Python 3.8+

Required Libraries:

+ scikit-learn

+ numpy

+ pandas

+ nltk

+ tensorflow
# Dataset used
The dataset used for this project is sourced from Kaggle and contains 9,900 labeled news articles.
We perform data preprocessing including:
 Removing stopwords
 Lemmatization & stemming
 TF-IDF vectorization
 # Model Training and Challenges Faced
 ## Model used
 For fake news classification, multiple machine learning models were trained, including:

+ Random Forest Classifier

+ Naïve Bayes Classifier

+ LSTM Neural Network

## Training Process
### Data Preprocessing:

Tokenization, stopword removal, stemming, and lemmatization were applied.

TF-IDF vectorization was used to convert text data into numerical features.

### Model Training:

The dataset was split into training (80%) and testing (20%) sets.

Each model was trained using the processed text data and optimized using hyperparameter tuning.

## Challenges Faced
### Class Imbalance:
The dataset had an unequal distribution of real and fake news, affecting model performance. SMOTE and other resampling techniques were explored to address this.

### Feature Engineering:
Selecting the right features significantly impacted model accuracy. TF-IDF worked well for traditional models, while word embeddings were better suited for deep learning.

### Computational Cost:
Training the LSTM model required higher computational power compared to traditional models. GPU acceleration was used to speed up training.

### Overfitting: 
Some models, especially deep learning ones, initially showed overfitting. Dropout layers and regularization techniques were applied to mitigate this issue.

# Model Performance and Visualization
We use multiple visualizations to evaluate the model's performance:

 + Confusion Matrix – Accuracy measurement
 + Feature Importance Graph – Identifies key words in classification
 + Word Cloud – Highlights frequent words in Fake/Real news
 + Learning Curves – Monitors model overfitting

# About
Fake News Detection using NLP & AI

### Topics Covered:

+  Text Classification

+  Machine Learning (Random Forest, Logistic Regression, SVM)

+ Deep Learning (LSTM, Word Embeddings)

+  Data Visualization

 


