#Analysis of precision and accuracy of the model:

# Load and preprocess all images in the dataset
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
all_images = []
all_labels = []
real_images_dir=r'F:\training_data_set\real'
fake_images_dir=r'F:\training_data_set\fake'
for filename in os.listdir(real_images_dir):
    img_path = os.path.join(real_images_dir, filename)
    img = preprocess_image(img_path, target_size)
    all_images.append(img)
    all_labels.append('real')

for filename in os.listdir(fake_images_dir):
    img_path = os.path.join(fake_images_dir, filename)
    img = preprocess_image(img_path, target_size)
    all_images.append(img)
    all_labels.append('fake')

# Predict labels for all images
predicted_labels = []
for img in all_images:
    discriminator_probability = loaded_discriminator.predict(img)[0][0]
    reconstructed_image = loaded_autoencoder.predict(img)
    mse = np.mean(np.square(img - reconstructed_image))
    
    if discriminator_probability < discriminator_threshold and mse > autoencoder_threshold:
        predicted_labels.append('fake')
    else:
        predicted_labels.append('real')

# Calculate metrics
precision = precision_score(all_labels, predicted_labels, average='weighted')
accuracy = accuracy_score(all_labels, predicted_labels)
recall = recall_score(all_labels, predicted_labels, average='weighted')
f1 = f1_score(all_labels, predicted_labels, average='weighted')

# Print metrics
print('Precision:', precision)
print('Accuracy:', accuracy)
print('Recall:', recall)
print('F1-Score:', f1)