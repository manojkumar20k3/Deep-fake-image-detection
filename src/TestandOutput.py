#Test the model with an image :

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc


# Load the saved discriminator model
loaded_discriminator = load_model('gandiscmodel.keras')

# Load the saved autoencoder model
loaded_autoencoder = load_model('automodel.keras')

# Load and preprocess an image for classification
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Set the image path and target size
test_image_path = r'F:\ma.jpg'  # Replace with the path to the test image
target_size = (75, 75)

# Preprocess the test image
preprocessed_image = preprocess_image(test_image_path, target_size)

# Use the Discriminator to predict the probability
discriminator_probability = loaded_discriminator.predict(preprocessed_image)[0][0]

# Use the Autoencoder to reconstruct the image
reconstructed_image = loaded_autoencoder.predict(preprocessed_image)

# Calculate the mean squared error (MSE) between the original and reconstructed image
mse = np.mean(np.square(preprocessed_image - reconstructed_image))


# Define the thresholds for both models
discriminator_threshold = optimal_threshold  # Set the threshold based on ROC curve
autoencoderthreshold = autoencoder_threshold  # Set the threshold based on the reconstruction error


# Classify the image using both models
if discriminator_probability < discriminator_threshold and mse > autoencoderthreshold:
    classification_result = 'fake'
else:
    classification_result = 'real'

# Print the classification result
print(discriminator_threshold)
print(autoencoderthreshold)
print('Image Classification:', classification_result)
print('Discriminator Probability:', discriminator_probability)
print('Autoencoder MSE:', mse)