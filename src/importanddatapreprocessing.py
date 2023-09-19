import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape, Dropout, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2

# Set paths to real and fake image directories
# replace with the paths to your original training data
real_images_dir = r'F:\training_data_set\real' 
fake_images_dir = r'F:\training_data_set\fake' 
genuine_images_dir = r'F:\training_data_set\real'

# Set the desired image size
target_size = (75, 75)

# Load and preprocess real and fake images
def load_and_preprocess_images_from_folder(folder_path, label):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            img = img.astype(np.float32) / 255.0
            images.append(img)
    labels = [label] * len(images)
    return np.array(images), np.array(labels)

real_images, real_labels = load_and_preprocess_images_from_folder(real_images_dir, label=1)
fake_images, fake_labels = load_and_preprocess_images_from_folder(fake_images_dir, label=0)
x_real = np.array(real_images)
x_fake = np.array(fake_images)
y_fake = np.array(fake_labels)
y_real = np.array(real_labels)

