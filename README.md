# Project: Classification of Rice Using CNN
# Datasource
The dataset for this project is sourced from Kaggle:
https://www.kaggle.com/code/nmaleknasr/rice-image-classification-cnn-tensorflow

# Objective
To develop and optimize a deep learning-based rice image classification model using TensorFlow. 
The project aims to accurately identify and categorize different types of rice grains using 
Convolutional Neural Networks (CNNs) to process visual features and classify them into predefined categories.

# Step-by-Step Flow for the Project
- Build the CNN Model
Loaded and preprocessed the rice image dataset with 75,000 images across five rice varieties.
Designed and trained multiple CNN models, starting with test models to explore configurations.
Evaluated model performance using metrics such as accuracy and validation loss.
Saved the optimized model for deployment and testing.

- Create a Flask API
Built a Flask-based web application for real-time testing.
Allowed users to upload rice images and receive predictions with confidence scores.
Integrated the trained CNN model into the Flask backend.

- Visualize Results in Tableau
Exported prediction data (e.g., class, accuracy) as CSV files.
Created interactive visualizations in Tableau, including:
Class distribution.
Model performance metrics.
Trends in confidence scores over time.
Identification of potential model drift and class imbalances.

- Store Predictions in SQL
Designed a SQL database schema to store prediction data.
Stored each prediction with details like image name, predicted class, confidence score, and timestamp.
Used Python's SQLAlchemy library to query and analyze model performance metrics.

# Dataset Features
Total Images: 75,000
Categories: Five rice varieties: Arborio, Basmati, Ipsala, Jasmine, and Karacadag.
Images were resized and normalized for faster computation and accurate predictions.

# CNN Model: Overview of Layers
- Convolution Layer: Extracts features like edges and textures from images.
-Pooling Layer: Reduces spatial dimensions to retain important features while improving computational efficiency.
-Flattening Layer: Converts feature maps into a single-dimensional vector.
-Fully Connected Layer: Connects features to output predictions.
-Output Layer: Outputs probabilities for the five rice classes.

# Test Models and Results
# Test Model 1
-Data Split: 50% training and 50% validation for balanced evaluation.
-Activation Functions:
-ReLU in convolutional layers for efficient training.
-Sigmoid in dense layers for simplicity.
- Epochs: Limited to 4 for faster experimentation.
- Input Size: 32 × 32 for reduced computational cost.
# Performance:
Training Accuracy: 96.81%
Validation Accuracy: 89.22%

# Test Model 2
-Data Split: 60% training and 40% validation for improved learning.
-Activation Functions:
-Sigmoid in convolutional layers for alternative feature extraction.
-Softmax in the output layer for probabilistic predictions.
-Epochs: Increased to 6 for better learning.
-Input Size: Same as Test Model 1 (32 × 32).
# Performance:
Training Accuracy: 97.01%
Validation Accuracy: 86.72%

# Optimized CNN Model
-Enhancements Over Test Models
-Data Augmentation:
-Added techniques like rotation, shifting, shearing, zooming, and horizontal flipping to improve generalization.
-Input Size:
Increased from 32 × 32 to 250 × 250 for detailed feature extraction.
-Model Architecture:
-Added a second convolutional layer with 64 filters.
-Increased dense layer units from 32 to 128.
-Used ReLU activation in all layers except the output layer, where Softmax was applied.
-Training Parameters:
-Increased epochs from 6 to 10 for better accuracy.
-Data Split: Adjusted to 80% training and 20% validation.
# Performance Improvement
Training Accuracy: 98.52%
Validation Loss: Reduced from 0.0943 to 0.0499.

# Flask API
-Developed a web application for classifying rice varieties in real time.
- Users can upload rice images, and the model predicts the class with a confidence score.
Example: Achieved a confidence score of 0.97578 for Basmati rice in one instance.

# Tableau Visualizations
-Insights from Visualizations
-Line Chart:
-Showed consistent model performance over time, with peaks reflecting testing phases.
-Box Plot:
-Highlighted model confidence levels, with narrower ranges indicating stable predictions.
-Outliers suggested potential areas for model improvement.

# Class Distribution:
-Showed imbalances in the dataset, emphasizing the need for careful tuning to avoid overfitting.
- Scatter plot revealed fluctuations in model confidence, aiding in monitoring and refinement.
  
# SQL Storage and Analysis
- Predictions were stored in a SQL database with details such as image name, predicted class, confidence score, and timestamp.
-Queried for:Average precision, recall, and F1-score to evaluate overall model performance.
-Identified underperforming categories with F1-scores below 0.98.

# Conclusion
This project demonstrates the iterative process of building and optimizing a CNN model for rice classification.
By combining deep learning with Flask, Tableau, and SQL, we achieved a robust pipeline for accurate 
predictions, insightful visualizations, and effective data management.
