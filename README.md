# Cat-Dog-classification
Tensorflow based classification CNN model


# Cats vs Dogs Classification

This project aims to classify images of cats and dogs using deep learning techniques. The dataset used for this project is publicly available from Kaggle: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data).

## Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

Install the required packages using the following command:

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn
```


# Cats vs Dogs Classification

This project aims to classify images of cats and dogs using deep learning techniques. The dataset used for this project is publicly available from Kaggle: [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data).

## Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Scikit-learn

Install the required packages using the following command:

```bash
pip install tensorflow numpy opencv-python matplotlib seaborn scikit-learn
```

# 1. Data Preprocessing:

The code unzips the 'train' and 'test1' folders, loads and preprocesses the data, and checks the dataset sizes.

# 2. Importing Packages:

Various packages are imported to facilitate data manipulation, model creation, training, and evaluation. This includes TensorFlow, OpenCV, NumPy, Scikit-learn, and more.

# 3. Model Creation:

The code defines a model creation function (create_model) that constructs a deep learning model for the classification task. The model includes data augmentation layers, convolutional layers, pooling layers, dropout, and dense layers.

# 4. Training:

A function (train_model) trains the defined model using the loaded and preprocessed data. It specifies the number of epochs, batch size, and other parameters for training.

# 5. Evaluation:

An evaluation function (evaluate_model) assesses the trained model's performance on validation data. It calculates metrics such as accuracy, precision, recall, and generates a confusion matrix and ROC curve.

# 6. Results:

The code loads, preprocesses, trains, and evaluates the model. It then prints the accuracy, precision, and recall values. Additionally, it plots the confusion matrix and the ROC curve to visualize the model's performance.

# Note

You'll need the necessary datasets to run this code. Ensure you have the 'train.zip' and 'test1.zip' files available.

Data preprocessing, model creation, and training functions are defined to promote code modularity and maintainability.

The RandomFlip, RandomRotation, and RandomZoom layers in data augmentation help prevent overfitting.



# Getting Started

Clone this repository to your local machine:
bash
```bash
git clone <https://github.com/HsuehYiLu/Cat-_Dog-_classification.git>
```


Navigate to the project directory:
bash
```bash
cd cats-vs-dogs-classification
```

Download the dataset from Kaggle and place it in the appropriate directories: https://www.kaggle.com/competitions/dogs-vs-cats/data.

Unzip the train and test1 folder.

Place the training images in data/train

Place the test images in data/test

Run the main script to train the model and evaluate its performance:


# Results

The trained model's performance is evaluated based on the following metrics:

Accuracy: 0.8428

Precision: 0.8092

Recall: 0.8994


Confusion Matrix

<img src="Result/CM.png">

ROC Curve

<img src="Result/ROC.png">


# License

This project is licensed under the MIT License.

