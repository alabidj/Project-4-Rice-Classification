Project Proposal: Classification of Rice using CNN

Datasource: https://www.kaggle.com/code/nmaleknasr/rice-image-classification-cnn-tensorflow

Objective statement:
To develop a deep learning-based rice image classification model using TensorFlow
that can accurately identify and categorize different types of rice grains from 
images. The project aims to leverage convolutional neural networks (CNNs) to 
process visual features of rice grains and classify them into predefined categories 

Step-by-Step Flow for Your Project:

Build the CNN Model:
Load and preprocess the rice image dataset.
Design and train the CNN model for rice classification.
Evaluate the model’s performance using metrics like accuracy.
Save the trained model for later use.

Create a Flask API:
Set up a Flask application.
Load the trained model in the Flask backend.
Create an endpoint to allow users to upload images.
Use the model to predict the rice type and return the result as a response.

Visualize Results in Tableau:
Export prediction data (e.g., class, accuracy) as a CSV file.
Design visualizations to display insights:
Class distribution of predictions.
Performance metrics and trends.
Create dashboards to summarize findings.

Store Predictions in SQL:
Design a SQL database schema to store prediction data.
Save each prediction with details like image name, predicted class, confidence score, and timestamp.
Retrieve stored predictions for analysis or reusability.


**Project Details**
Dataset Features:
75,000 images of 5 different types of Rice

CNN Model:
CNN models were built with the fundamental layers commonly used in image classification tasks.
Convolution Layer: Extracts features like edges and textures.
Pooling Layer: Reduces spatial dimensions for computational efficiency.
Flattening Layer: Prepares data for dense layers by converting feature maps into a vector.
Fully Connected Layer: Connects features to output predictions, using Sigmoid activation for non-linear transformation.
Output Layer: Final dense layer with Sigmoid activation for multi-class output.

Test Model 1 Results:
1-Data Split: A 50-50% split between training and validation was chosen to evenly test the model's performance.
2-Activation Functions: `ReLU was used in the convolutional layers for its efficiency in training, while Sigmoid was used in the dense layers for simplicity.
3-Epochs- Limited to 4 for quick testing without overfitting.
4-Input Size- Reduced to (32, 32) for faster computation during training.

Test Model 2 Results
1-Data Split- A larger portion (60%) of the dataset is used for training and 40% validation
2-Activation Functions: Sigmoid in the convolutional layer to explores an alternative to the commonly used ReLU for feature extraction. 
3-Epochs- Training for 6 epochs ensures the model has sufficient iterations to learn patterns in the data.
4-Input Size- This remains same as previous model.

Optimized CNN Model:
The model is created using TensorFlow's Sequential class, which allows layers to be added in a linear stack.

For this model the data was split into 80% for the data training and 20% for data testing.

A Convolutional Layer with 32 filters and a 3x3 kernel.
A Pooling Layer (MaxPooling) after the first convolutional layer.
A Second Convolutional Layer with 64 filters and a 3x3 kernel.
A Flattening Layer to prepare data for the fully connected layers.
A Dense Layer with 128 neurons.
An Output Layer with 5 neurons (for the number of classes).

-Data Augmentation- Introduced augmentation techniques: rotation, shifting, shearing, zooming, and horizontal flipping for better generalization. The data split was changed to 80-20% for this model.

2-Input Image Size:Increased from 32 × 32 to 250 × 250 for detailed feature extraction.

3-Model Architecture- Added a second convolutional layer with 64 filters.
Increased the dense layer units from 32 to 128.
Activation Functions: Used ReLU in all layers except the output layer for better gradient flow and training efficiency. For the output layer softmax activation was used.

4-Training Parameters: Increased epochs from 6 to 10 to achieve more accuracy.

Performance Improvement:Accuracy improved from 97.01% to 98.52%.
Validation loss reduced from 0.0943 to 0.0499.

Flask API:
Built a web application for classifying different types of rice using a trained Convolutional Neural Network (CNN).
Users upload rice images, and the app returns the predicted class with confidence.

Confidence of 0.97578 was achieved for one instance of Basmati rice

Tableau Visualizations Analysis 1 (Line chart): 
Downward trend suggests a reduction in usage or delays in processing 
Consistent activity overtime reflects a steady model performance and usage
Peaks in activity may reflect testing phases or specific events 

Tableau Visualizations Analysis 2 (Box plot): 

Higher medians and narrower ranges suggest the model is confident in its predictions for specific classes
Classes with wide ranges or lower medians might require additional model tuning or data augmentation 
Outliers in confidence scores can indicate predications that deviate significantly from the norm 

SQl Storage:
Using Python's SQLAlchemy library, the classification report was inserted into the database.
The first query was used to retrieve all rows from the classification_report table.
To understand the model's overall performance, average precision, recall, and F1-score were calculated.
To identify under performing categories, classes with an F1-score below a threshold (0.98) were queried:
