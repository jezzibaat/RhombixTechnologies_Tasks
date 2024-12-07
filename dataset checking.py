import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models

# Define parameters
img_height = 224
img_width = 224
batch_size = 32
epochs = 10

# Function to load and preprocess images
def load_and_preprocess_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_height, img_width])
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Function to get labels from file paths
def get_label(file_path):
    return tf.strings.split(file_path, os.sep)[-2]

# Load the dataset
dataset_dir = 'PlantVillage'
sample_file_paths = tf.data.Dataset.list_files('PlantVillage/*/*.jpg')

# Shuffle and split the dataset
num_files = sum(1 for _ in sample_file_paths)
train_size = int(0.8 * num_files)  # 80% for training
val_size = int(0.1 * num_files)    # 10% for validation
test_size = num_files - train_size - val_size  # 10% for testing

# Shuffle the dataset
sample_file_paths = sample_file_paths.shuffle(buffer_size=num_files)

# Create training, validation, and test datasets
train_dataset = sample_file_paths.take(train_size)
val_dataset = sample_file_paths.skip(train_size).take(val_size)
test_dataset = sample_file_paths.skip(train_size + val_size)

# Get unique class names and create a mapping
class_names = np.unique([get_label(file_path).numpy().decode('utf-8') for file_path in train_dataset])
class_indices = {name: index for index, name in enumerate(class_names)}

# Create a lookup table for class indices
keys = tf.constant(list(class_indices.keys()))
values = tf.constant(list(class_indices.values()))
lookup_table = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys, values), default_value=-1)

# Map the labels to the datasets
def get_numeric_label(file_path):
    label = get_label(file_path)
    return lookup_table.lookup(label)

train_labels = train_dataset.map(get_numeric_label)
val_labels = val_dataset.map(get_numeric_label)
test_labels = test_dataset.map(get_numeric_label)

# Create a dataset of images and labels
train_dataset = tf.data.Dataset.zip((train_dataset.map(load_and_preprocess_image), train_labels))
val_dataset = tf.data.Dataset.zip((val_dataset.map(load_and_preprocess_image), val_labels))
test_dataset = tf.data.Dataset.zip((test_dataset.map(load_and_preprocess_image), test_labels))

# Load a pre-trained model
base_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

# Freeze the base model
base_model.trainable = False

# Create a new model on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Prepare the datasets for training
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Train the model
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

# Prepare the test dataset
test_dataset = test_dataset.batch(batch_size)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f'Test accuracy: {test_accuracy:.2f}')

# Function to predict a single image
def predict_image(file_path):
    img = load_and_preprocess_image(file_path)
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Example usage of prediction
predicted_class = predict_image(r'C:\Users\jk537\Downloads\Data Sciencework\Internships\Rhombix Technologies!!\MONTH 3\PlantVillage\Tomato_healthy\7a5fe2a0-ac85-4a37-b0f6-8cfc8fb21c9c___RS_HL 9865.jpg')
print(f'Predicted class: {predicted_class}')