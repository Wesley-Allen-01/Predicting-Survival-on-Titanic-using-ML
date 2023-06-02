# Predicting-Survival-on-Titanic-using-ML

This project focuses on using machine learning techniques to predict the survival of passengers on the Titanic. The goal is to develop experience with the TensorFlow library and enhance data manipulation skills. The project utilizes the standard Titanic dataset from Seaborn and trains a Sequential Neural Network on the manipulated data. The model achieved an impressive 99% accuracy score when tested on unseen data from the dataset.

## Libraries Used
The project makes use of the following libraries:

numpy: for data handling
pandas: for data manipulation
seaborn: for data visualization
sklearn: for data preprocessing and model evaluation
tensorflow and keras: for developing the neural network model
##Dataset Exploration
The code begins by importing necessary libraries and loading the Titanic dataset using Seaborn's built-in function sns.load_dataset('titanic'). The titanic.head() function is used to display the first few rows of the dataset, while titanic.info() and titanic.describe() provide information about the dataset's structure and summary statistics. The print(titanic.columns) statement displays the names of all columns in the dataset.

## Data Preparation
To prepare the data for the neural network, several steps are taken. First, missing values in the 'age' column are filled with the median age of the dataset using the fillna() function. Next, categorical variables are converted into dummy variables using one-hot encoding with the help of pd.get_dummies() function. The 'survived' column is separated as the target variable 'y', and the rest of the dataset is stored in 'X'. The dataset is then split into training and test subsets using the train_test_split() function from sklearn. The data is also scaled using the StandardScaler() class to have zero mean and unit variance.

## Model Creation and Training
The code defines a Sequential model using keras.Sequential() and adds three dense layers to it. The first two layers have 64 neurons each and use the ReLU activation function, while the last layer has one neuron and uses the sigmoid activation function. The model is then compiled with the Adam optimizer, binary cross-entropy loss function, and accuracy as the metric for evaluation. The model.fit() function is used to train the model on the scaled training data for 15 epochs, with a validation split of 20% of the training data.

## Model Evaluation
After training, the model is evaluated on the test data. The predictions for the test data are obtained using model.predict() and a threshold of 0.5 to classify the predictions as either 0 or 1. The accuracy of the model is calculated using accuracy_score() from sklearn, and the result is printed. Additionally, a confusion matrix is created using confusion_matrix() from sklearn and visualized using a heatmap from Seaborn.

## Summary
The model achieved an accuracy of 99% on the test data, indicating that it correctly predicted the survival of passengers on the Titanic. The project highlights the successful implementation of machine learning techniques using TensorFlow and showcases the data manipulation skills required for such tasks. The creator expresses satisfaction with the model's performance and looks forward to future projects and further skill development in this field.
