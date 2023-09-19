# Build the Autoencoder model
latent_dim = 32

input_shape = (target_size[0], target_size[1], 3)
inputs = Input(shape=input_shape)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
encoded = Dense(latent_dim, activation='relu')(x)

x = Dense(19 * 19 * 32, activation='relu')(encoded)
x = Reshape((19, 19, 32))(x)
x = Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
x = Dropout(0.2)(x)
x = Conv2DTranspose(3, (3, 3), strides=(1, 1), activation='sigmoid', padding='same')(x)
decoded = tf.image.resize(x, target_size, method='bilinear')

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the Autoencoder on real images
autoencoder.fit(x_real, x_real, epochs=400, batch_size=128, shuffle=True)

# Save the trained Autoencoder model
autoencoder.save('automodel.keras')

# Calculate the reconstruction error-based threshold
genuine_mses = []
for filename in os.listdir(genuine_images_dir):
    img_path = os.path.join(genuine_images_dir, filename)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    reconstructed_img = autoencoder.predict(img)
    mse = np.mean(np.square(img - reconstructed_img))
    genuine_mses.append(mse)

average_mse = np.mean(genuine_mses)
stddev_mse = np.std(genuine_mses)
threshold_multiplier = 2
autoencoder_threshold = average_mse + threshold_multiplier * stddev_mse

# Build the Discriminator model
def build_discriminator(latent_dim, image_shape):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=image_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

discriminator = build_discriminator(latent_dim, (75, 75, 3))
discriminator.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

# Train the GAN model
def build_generator(latent_dim, image_shape):
    model = tf.keras.Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization())
    model.add(Dense(np.prod(image_shape), activation='tanh'))
    model.add(Reshape(image_shape))
    return model

def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    model = Model(gan_input, gan_output)
    return model

generator = build_generator(latent_dim, (75, 75, 3))
gan = build_gan(generator, discriminator)
gan.compile(optimizer=Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

epochs = 10000
batch_size = 128

for epoch in range(epochs):
    # Generate random noise for the generator input
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))

    # Generate fake images using the generator
    generated_images = generator.predict(noise)

    # Select a random batch of real images from the dataset
    idx = np.random.randint(0, x_real.shape[0], batch_size)
    real_images_batch = x_real[idx]

    # Label real and fake images for the discriminator
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_images_batch, valid)
    d_loss_fake = discriminator.train_on_batch(generated_images, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    noise = np.random.normal(0, 1, size=(batch_size, latent_dim))
    valid = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid)

    print(f"Epoch: {epoch+1}/{epochs}, D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
    
    

# Use the Discriminator to predict probabilities of being real
discriminator_probabilities_real = discriminator.predict(x_real)
discriminator_probabilities_fake = discriminator.predict(x_fake)

# Combine real and fake probabilities for ROC curve
all_probabilities = np.concatenate((discriminator_probabilities_real, discriminator_probabilities_fake), axis=0)
all_labels = np.concatenate((np.ones_like(discriminator_probabilities_real), np.zeros_like(discriminator_probabilities_fake)), axis=0)

# Compute the ROC curve and AUC
fpr, tpr, thresholds = roc_curve(all_labels, all_probabilities)
roc_auc = auc(fpr, tpr)

# Find the optimal threshold based on the ROC curve
optimal_threshold_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_threshold_idx]

print(f"Optimal Threshold: {optimal_threshold:.3f}")

# Save the trained Discriminator model
discriminator.save('gandiscmodel.keras')

# Use the Autoencoder to reconstruct real and fake images
reconstructed_real_images = autoencoder.predict(x_real)
reconstructed_fake_images = autoencoder.predict(x_fake)

# Calculate reconstruction errors
mse_real = np.mean(np.square(x_real - reconstructed_real_images), axis=(1, 2, 3))
mse_fake = np.mean(np.square(x_fake - reconstructed_fake_images), axis=(1, 2, 3))

# Classify images using both Discriminator and Autoencoder
classification_results = []

for i in range(len(x_real)):
    if discriminator_probabilities_real[i] > optimal_threshold and mse_real[i] < autoencoder_threshold :
        classification_results.append('Real')
    else:
        classification_results.append('Fake')

for i in range(len(x_fake)):
    if discriminator_probabilities_fake[i] > optimal_threshold and mse_fake[i] < autoencoder_threshold:
        classification_results.append('Real')
    else:
        classification_results.append('Fake')

# Print the classification results
for i, result in enumerate(classification_results):
    print(f"Image {i+1}: {result}")






