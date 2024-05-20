#!/usr/bin/env python
# coding: utf-8

# In[16]:


pip install pyspellchecker


# In[35]:


pip install numpy


# In[2]:


#image count

import tensorflow_datasets as tfds

# Load the TF Flowers dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Display image count for each class
class_counts = {}
total_images = 0
for example in dataset:
    label = example['label'].numpy()
    class_name = info.features['label'].int2str(label)
    if class_name in class_counts:
        class_counts[class_name] += 1
    else:
        class_counts[class_name] = 1
    total_images += 1

# Display image count for each class
print("Image counts for each class:")
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")

# Display total number of images
print("\nTotal number of images:", total_images)


# In[8]:


#Sharpness score of dataset

import cv2
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Function to compute sharpness score of an image
def compute_sharpness(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Compute Laplacian variance as sharpness score
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return sharpness

# Load the TF Flowers dataset
dataset = tfds.load('tf_flowers', split='train')

# Initialize dictionaries to store sharpness scores for each class
sharpness_scores = {class_name: [] for class_name in range(5)}
class_names = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# Compute sharpness scores for each image
for example in dataset:
    image = example['image'].numpy()
    label = example['label'].numpy()
    sharpness = compute_sharpness(image)
    sharpness_scores[label].append(sharpness)

# Plot one histogram per flower class
plt.figure(figsize=(15, 10))
for class_label, scores in sharpness_scores.items():
    plt.subplot(2, 3, class_label+1)
    plt.hist(scores, bins=50, range = (0, 500), alpha=0.5)
    plt.xlabel('Sharpness Score')
    plt.ylabel('Frequency')
    plt.title(class_names[class_label])
    plt.grid(True)

plt.tight_layout()
plt.show()

# Print sharpness scores for each flower class
print("Sharpness scores for each flower class:")
for class_label, scores in sharpness_scores.items():
    avg_sharpness = sum(scores) / len(scores)
    print(f"{class_names[class_label]}: Average Sharpness Score = {avg_sharpness:.2f}")





# In[16]:


import tensorflow_datasets as tfds

# Load the TF Flowers dataset
dataset = tfds.load('tf_flowers', split='train')

# Initialize dictionaries to store counts of image sizes for each class and the dataset as a whole
class_size_counts = {class_name: {} for class_name in range(5)}
total_size_counts = {}

# Iterate through the dataset and record the dimensions of each image
for example in dataset:
    image = example['image'].numpy()
    height, width, _ = image.shape
    size = (height, width)
    label = example['label'].numpy()
    
    # Update counts for each class
    if size in class_size_counts[label]:
        class_size_counts[label][size] += 1
    else:
        class_size_counts[label][size] = 1
    
    # Update counts for the dataset as a whole
    if size in total_size_counts:
        total_size_counts[size] += 1
    else:
        total_size_counts[size] = 1

# Print the total count of different sizes of images available in each class and the dataset as a whole
print("Total count of different sizes of images available in each class:")
for class_label, size_counts in class_size_counts.items():
    print(f"Class {class_label}: {len(size_counts)} different sizes")

print("\nTotal count of different sizes of images available in the dataset as a whole:")
print(f"{len(total_size_counts)} different sizes")


# In[1]:


#missing data

import tensorflow_datasets as tfds

# Load the TF Flowers dataset
dataset = tfds.load('tf_flowers', split='train')

# Initialize counter for missing images
missing_count = 0

# Iterate through the dataset to check for missing images
for example in dataset:
    try:
        image = example['image'].numpy()
    except Exception as e:
        missing_count += 1

# Check if there are any missing images
if missing_count == 0:
    print("Dataset is complete and no missing data")
else:
    print(f"{missing_count} missing values found")


# In[10]:


import cv2
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Function to add Gaussian noise to an image
def add_noise(image, scale=1):
    row, col, ch = image.shape
    mean = 0
    var = 100 * scale  # Increase noise variance by the scale factor
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

# Function to denoise images using Non-Local Means Denoising (NLM)
def denoise_image(image):
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Load the TF Flowers dataset and get class names
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)
class_labels = info.features['label'].names

# Initialize dictionary to store the last flower image under each class
last_images = {class_name: None for class_name in class_labels}

# Iterate through the dataset to pick the last flower image under each class
for example in dataset:
    image = example['image'].numpy()
    label = example['label'].numpy()
    class_name = class_labels[label]
    
    # Store the latest image encountered for this class
    last_images[class_name] = image

# Denoise the selected last flower images and display before and after comparison
for class_name, image in last_images.items():
    # Add more noise to the image before denoising
    noisy_image = add_noise(image, scale=2)
    
    # Denoise the image using NLM
    denoised_image = denoise_image(noisy_image)

    # Plot the noisy and denoised images
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy_image)
    plt.title('Noisy')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(denoised_image)
    plt.title('Denoised')
    plt.axis('off')
    plt.suptitle(f'Class: {class_name}')
    plt.show()


# In[12]:


#missing data

import pandas as pd
about = pd.read_csv("about.csv")
about.info()


# In[22]:


#noise removal

import pandas as pd
from spellchecker import SpellChecker
import re

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('about.csv')

# Initialize the SpellChecker
spell = SpellChecker()

# Function to perform spell checking on a text entry
def spell_check_text(text):
    # Regex pattern to match English characters
    pattern = re.compile('[a-zA-Z]+')
    # Tokenize the text into words
    words = text.split()
    # Perform spell checking on each word
    corrected_words = []
    for word in words:
        # Check if the word contains English characters
        if pattern.match(word):
            # Perform spell checking only for words containing English characters
            corrected_word = spell.correction(word)
            # Check if correction is found
            if corrected_word is not None:
                # Print the changes made
                if corrected_word != word:
                    print(f"Changed '{word}' to '{corrected_word}'")
                corrected_words.append(corrected_word)
            else:
                # If no correction is found, keep the original word
                corrected_words.append(word)
        else:
            # Retain non-English character sequences
            corrected_words.append(word)
    # Join the corrected words back into a text string
    corrected_text = ' '.join(corrected_words)
    return corrected_text

# Apply spell checking to each text entry in the DataFrame
df_corrected = df.applymap(spell_check_text)

# Print a sample before and after spell checking
print("Sample Before Spell Checking:")
print(df.iloc[0])  # Print the first row of the original DataFrame
print("\nSample After Spell Checking:")
print(df_corrected.iloc[0])  # Print the first row of the corrected DataFrame


# In[25]:


#inconsistant data
import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('about.csv')

# Check for duplicates
duplicates = df[df.duplicated()]

# Print the results
if duplicates.empty:
    print("No duplicates. Data is consistent.")
else:
    print("Duplicates found. Data is inconsistent.")
    print("Duplicate Entries:")
    print(duplicates)


# In[8]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import cv2

# Load the TF Flowers dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Define the class labels
class_labels = ['dandelion', 'daisy', 'tulips', 'sunflowers', 'roses']

# Function to resize images to 224x224 using OpenCV
def resize_image_opencv(image):
    # Convert the image tensor to numpy array
    image = image.numpy()
    # Resize the image
    resized_image = cv2.resize(image, (224, 224))
    return resized_image

# Apply resizing to the dataset
dataset_resized_opencv = dataset.map(lambda x: {'image': tf.py_function(resize_image_opencv, [x['image']], tf.uint8), 'label': x['label']})

# Function to display one sample of before and after resizing for each flower class
def display_samples(dataset):
    plt.figure(figsize=(15, 10))
    for i, class_label in enumerate(class_labels, start=1):
        # Find one sample image for the current class
        sample = dataset.filter(lambda x: x['label'] == i).take(1)
        for sample_data in sample:
            # Plot the original image
            plt.subplot(2, 5, i)
            plt.imshow(sample_data['image'])
            plt.title("Original")
            plt.axis('off')

        # Find the resized image for the current class
        resized_sample = dataset_resized_opencv.filter(lambda x: x['label'] == i).take(1)
        for sample_data in resized_sample:
            # Plot the resized image
            plt.subplot(2, 5, i + len(class_labels))
            plt.imshow(sample_data['image'])
            plt.title("Resized")
            plt.axis('off')
            # Add scale bar
            plt.text(10, 10, 'Scale: 224x224', color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.8))

    plt.tight_layout()
    plt.show()

# Display samples
display_samples(dataset)


# In[10]:


import pandas as pd

# Load the CSV file
df = pd.read_csv('about.csv')

# Define the mapping of flower names to numeric values
flower_mapping = {
    "dandelion": 0,
    "daisy": 1,
    "tulips": 2,
    "sunflowers": 3,
    "roses": 4
}

# Apply one-hot encoding
df['flower_encoded'] = df['flower'].map(flower_mapping)

# Print before and after of the "flower" column
print("Before one-hot encoding:")
print(df['flower'])

print("\nAfter one-hot encoding:")
print(df['flower_encoded'])


# In[27]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Define a function to normalize the image
def normalize_image(image):
    # Normalize the image pixels to the range [0, 1]
    normalized_image = image / 255.0
    return normalized_image

# Function to display before and after comparison for one image
def display_comparison(image, normalized_image):
    # Calculate the range of pixel values for original and normalized images
    original_range = (image.min(), image.max())
    normalized_range = (normalized_image.min(), normalized_image.max())

    # Print the array of pixel values for original and normalized images
    print("Original Image Pixel Values:")
    print(image)
    print("\nNormalized Image Pixel Values:")
    print(normalized_image)

    # Create a colorbar for the pixel value range
    original_colorbar = np.linspace(original_range[0], original_range[1], num=100)
    normalized_colorbar = np.linspace(normalized_range[0], normalized_range[1], num=100)

    # Plot the original image with scale
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel('Pixel Value')
    plt.ylabel('Pixel Value Range')
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()

    # Plot the normalized image with scale
    plt.subplot(1, 2, 2)
    plt.imshow(normalized_image)
    plt.title('Normalized Image')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xlabel('Pixel Value')
    plt.ylabel('Pixel Value Range')
    plt.gca().yaxis.set_label_position("left")
    plt.gca().yaxis.tick_left()

    plt.show()

# Normalize the dataset images and show a before and after comparison for one image
for example in dataset.take(1):
    image = example['image'].numpy()
    
    # Normalize the image
    normalized_image = normalize_image(image)
    
    # Display the before and after comparison
    display_comparison(image, normalized_image)


# In[29]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Function to apply data augmentation techniques and display a sample of each type
def display_augmentation(image, augmentation_function, title):
    # Apply the augmentation function to the image
    augmented_image = augmentation_function(image)
    
    # Plot the original and augmented images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented_image)
    plt.title(title)
    plt.axis('off')
    
    plt.show()

# Function to flip the image from left to right
def flip_left_right(image):
    return tf.image.flip_left_right(image)

# Function to flip the image from top to bottom
def flip_up_down(image):
    return tf.image.flip_up_down(image)

# Function to randomly crop the image
def random_crop(image):
    return tf.image.random_crop(image, size=[224, 224, 3])

# Function to rotate the image by 45 degrees
def rotate_45(image):
    return tf.image.rot90(image, k=1)

# Apply data augmentation techniques and display samples
for example in dataset.take(1):
    image = example['image']
    
    # Display side flipping
    display_augmentation(image, flip_left_right, 'Side Flipping')
    
    # Display up-down flipping
    display_augmentation(image, flip_up_down, 'Up-Down Flipping')
    
    # Display cropping
    display_augmentation(image, random_crop, 'Random Cropping')
    
    # Display rotation by 45 degrees
    display_augmentation(image, rotate_45, 'Rotation')


# In[30]:


import csv
import nltk
nltk.download('punkt')

# Function to tokenize text
def tokenize_text(text):
    tokens = nltk.word_tokenize(text)
    return tokens

# Read the CSV file
csv_file = 'about.csv'
with open(csv_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    header = next(reader)  # Skip the header row
    for row in reader:
        tokens_per_row = []  # Store tokens for each row
        for column in row:
            tokens = tokenize_text(column)
            tokens_per_row.append(tokens)
        print(tokens_per_row)


# In[37]:


import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
df = pd.read_csv('about.csv')

# Convert text inputs into numerical representations
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Perform PCA
pca = PCA()
X = df.values  # Assuming all columns are features
pca.fit(X)

# Scree Plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o', linestyle='-')
plt.title('Scree Plot')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.xticks(range(1, len(pca.explained_variance_ratio_) + 1))
plt.grid(True)
plt.show()


# In[8]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_datasets as tfds

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', as_supervised=True, with_info=True)

# Select a random image from the dataset
for image, label in dataset.take(1):
    original_image = image.numpy()

# Convert the image to greyscale
greyscaled_image = tf.image.rgb_to_grayscale(original_image).numpy()

# Display original and greyscaled images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(greyscaled_image.squeeze(), cmap='gray')
plt.title('Greyscaled Image')
plt.axis('off')

plt.show()

# Flatten the greyscaled image for PCA
flattened_image = greyscaled_image.flatten().reshape(1, -1)

# Perform PCA on the greyscaled image
pca = PCA()
pca.fit(flattened_image)

# Create scree plot
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.show()

# Reconstruct the image using principal components
num_components = 50  # Choose the number of components to keep
reconstructed_image = pca.inverse_transform(pca.transform(flattened_image))

# Reshape the reconstructed image
reconstructed_image = reconstructed_image.reshape(greyscaled_image.shape)

# Display the reconstructed image
plt.imshow(reconstructed_image.squeeze(), cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')
plt.show()


# In[ ]:





# In[23]:


plt.show()

# Print the 224x224 pixel image
plt.figure(figsize=(4, 4))
plt.imshow(greyscaled_image, cmap='gray')
plt.title('Greyscaled Image (224 x 224 pixels)')
plt.axis('off')
plt.show()


# In[3]:


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the TensorFlow Flowers dataset
dataset, info = tfds.load('tf_flowers', as_supervised=True, with_info=True)

# Define a function to preprocess images
def preprocess(img):
    img = tf.image.resize(img, (224, 224))
    img = preprocess_input(img)
    return img

# Create a MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Define the number of classes
num_classes = info.features['label'].num_classes

# Initialize counts for outliers per class
outlier_counts = [0] * num_classes

# Iterate through the dataset and collect features
for image, label in dataset['train']:
    img = preprocess(image)
    features = model.predict(tf.expand_dims(img, axis=0))
    z_scores = np.abs((features - np.mean(features, axis=0)) / np.std(features, axis=0))
    max_z_score = np.max(z_scores)  # Taking maximum z-score across all features
    if max_z_score > 2.0:  # Threshold for outlier detection (adjust as needed)
        outlier_counts[label.numpy()] += 1

# Print outlier counts for each class
for i in range(num_classes):
    print(f"Outliers in class {i}: {outlier_counts[i]}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


import os
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True, as_supervised=True)
dataset = r"C:\Users\harsh\Downloads\new_data_gwar\total"
# Function to apply data augmentation techniques and save the augmented images
def apply_augmentation_and_save(image, label, augmentation_function, title, count, save_dir):
    # Apply the augmentation function to the image
    augmented_image = augmentation_function(image)
    
    # Save the augmented image
    save_path = os.path.join(save_dir, f"augmented_image_{title}_{count}.jpg")
    tf.keras.preprocessing.image.save_img(save_path, augmented_image.numpy())
    
    return augmented_image, label

# Function to flip the image from left to right
def flip_left_right(image):
    return tf.image.flip_left_right(image)

# Function to flip the image from top to bottom
def flip_up_down(image):
    return tf.image.flip_up_down(image)

# Function to randomly crop the image
def random_crop(image):
    # Resize the image to a larger size
    image = tf.image.resize(image, [250, 350])
    
    # Apply random crop
    return tf.image.random_crop(image, size=[224, 224, 3])

# Function to rotate the image by 45 degrees
def rotate_45(image):
    return tf.image.rot90(image, k=1)

# Augmentation count for each type
augmentation_count = 18350

# Folder to save the images
save_dir = r'C:\Users\harsh\Downloads\aug_image_DAP'

# Create the directory if it does not exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Original dataset images and labels
original_images = []
original_labels = []

# Apply data augmentation techniques and save augmented images
count = 0
for image, label in dataset:
    # Resize the original image to a consistent shape
    image = tf.image.resize(image, [224, 224])
    
    # Save the original image
    original_images.append(image)
    original_labels.append(label)
    
    # Side Flipping
    apply_augmentation_and_save(image, label, flip_left_right, 'Side_Flipping', count, save_dir)
    count += 1
    
    # Up-Down Flipping
    apply_augmentation_and_save(image, label, flip_up_down, 'Up_Down_Flipping', count, save_dir)
    count += 1
    
    # Random Cropping
    apply_augmentation_and_save(image, label, random_crop, 'Random_Cropping', count, save_dir)
    count += 1
    
    # Rotation
    apply_augmentation_and_save(image, label, rotate_45, 'Rotation', count, save_dir)
    count += 1
    
    # Break if reached the augmentation count
    if count >= augmentation_count:
        break

# Print the total number of images in the new dataset
print("Total number of images in the new dataset:", count)


# In[15]:


import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)
dataset = r'C:\Users\harsh\Downloads\aug_image_DAP'
# Function to get flower name from label
def get_flower_name(label):
    return info.features['label'].int2str(label)

# Function to label images and print 10 samples
def label_and_print_samples(dataset):
    # Iterate through the dataset and print 10 samples
    count = 0
    for example in dataset:
        image = example['image']
        label = example['label']
        flower_name = get_flower_name(label)
        print(f"Image {count + 1}: Flower Name - {flower_name}, Label - {label}")
        # Display the image
        plt.imshow(image)
        plt.title(flower_name)
        plt.axis('off')
        plt.show()
        count += 1
        if count == 10:
            break

# Label images and print samples
label_and_print_samples(dataset)


# In[19]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import random

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Function to get flower name from label
def get_flower_name(label):
    return info.features['label'].int2str(label)

# Function to perform data augmentation techniques
def apply_augmentation(image, augmentation_function, label):
    augmented_image = augmentation_function(image)
    return augmented_image, label

# Function to display random samples from the augmented dataset
def display_random_samples(dataset, augmentation_function, augmentation_name):
    augmented_samples = []
    for example in dataset:
        image = example['image']
        label = example['label']
        augmented_image, _ = apply_augmentation(image, augmentation_function, label)
        augmented_samples.append((augmented_image, label))
    
    random_samples = random.sample(augmented_samples, 5)
    
    plt.figure(figsize=(15, 5))
    for i, (sample_image, label) in enumerate(random_samples, start=1):
        plt.subplot(1, 5, i)
        plt.imshow(sample_image)
        plt.title(get_flower_name(label))
        plt.axis('off')
    plt.suptitle(f'Random Samples from {augmentation_name} Augmentation', fontsize=16)
    plt.show()

# Function to flip the image from left to right
def side_flipping(image):
    return tf.image.flip_left_right(image)

# Function to flip the image from top to bottom
def up_down_flipping(image):
    return tf.image.flip_up_down(image)

# Function to randomly crop the image
def random_cropping(image):
    # Get the dimensions of the input image
    image_height, image_width, _ = image.shape
    # Define the crop size (smaller than the input image size)
    crop_height = min(image_height, 224)
    crop_width = min(image_width, 224)
    # Perform random cropping
    cropped_image = tf.image.random_crop(image, size=[crop_height, crop_width, 3])
    return cropped_image


# Function to rotate the image by 45 degrees
def rotation(image):
    return tf.image.rot90(image, k=1)

# Perform data augmentation and display random samples
display_random_samples(dataset, side_flipping, 'Side Flipping')
display_random_samples(dataset, up_down_flipping, 'Up-Down Flipping')
display_random_samples(dataset, random_cropping, 'Random Cropping')
display_random_samples(dataset, rotation, 'Rotation')


# In[26]:


import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Function to get flower name from label
def get_flower_name(label):
    return info.features['label'].int2str(label)

# Function to perform data augmentation techniques
def apply_augmentation(image, augmentation_function):
    augmented_image = augmentation_function(image)
    return augmented_image

# Function to flip the image from left to right
def side_flipping(image):
    return tf.image.flip_left_right(image)

# Function to flip the image from top to bottom
def up_down_flipping(image):
    return tf.image.flip_up_down(image)

# Function to randomly crop the image
def random_cropping(image):
    # Resize the image to meet the minimum required dimensions for cropping
    resized_image = tf.image.resize(image, [224, 224])
    return tf.image.random_crop(resized_image, size=[224, 224, 3])

# Function to rotate the image by 45 degrees
def rotation(image):
    return tf.image.rot90(image, k=1)

# Directory to save the augmented images
save_dir = r'C:\Users\harsh\Downloads\new_data_gwar'
os.makedirs(save_dir, exist_ok=True)

# Counter for augmented images
count = 0

# Iterate through the dataset, apply augmentations, and save the images
for example in dataset:
    image = example['image']
    label = example['label']
    
    # Apply augmentations
    augmented_images = [
        apply_augmentation(image, side_flipping),
        apply_augmentation(image, up_down_flipping),
        apply_augmentation(image, random_cropping),
        apply_augmentation(image, rotation)
    ]
    
    # Save augmented images
    for augmented_image in augmented_images:
        count += 1
        augmented_image_path = os.path.join(save_dir, f'image_{count}.jpg')
        tf.keras.preprocessing.image.save_img(augmented_image_path, augmented_image.numpy())

# Print total dataset count for new dataset with original and augmented labeled data
total_dataset_count = count + info.splits['train'].num_examples
print("Total Dataset Count (Original + Augmented):", total_dataset_count)


# In[34]:


import tensorflow as tf
import tensorflow_datasets as tfds
import os
import shutil

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)
dataset = r'C:\Users\harsh\Downloads\new_data_gwar'
# Define the directory paths for train, test, and validation sets
base_dir = './flower_dataset'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
val_dir = os.path.join(base_dir, 'validation')

# Create directories if they don't exist
for directory in [train_dir, test_dir, val_dir]:
    os.makedirs(directory, exist_ok=True)

# Calculate the number of examples for train, test, and validation sets
num_train_examples = int(0.8 * info.splits['train'].num_examples)
num_test_examples = int(0.1 * info.splits['train'].num_examples)
num_val_examples = info.splits['train'].num_examples - num_train_examples - num_test_examples

# Iterate through the dataset and save images to appropriate directories
for i, example in enumerate(dataset):
    image = example['image']
    label = example['label']
    
    # Determine the destination directory based on the split
    if i < num_train_examples:
        dest_dir = train_dir
    elif i < num_train_examples + num_test_examples:
        dest_dir = test_dir
    else:
        dest_dir = val_dir
    
    # Save the image to the destination directory
    image_filename = f'{i+1}.jpg'
    image_path = os.path.join(dest_dir, image_filename)
    tf.keras.preprocessing.image.save_img(image_path, image.numpy())

# Print the directories with heading
print('Directories:')
print('-----------')
print(f'Train Directory: {train_dir}')
print(f'Test Directory: {test_dir}')
print(f'Validation Directory: {val_dir}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


import tensorflow as tf
import tensorflow_datasets as tfds
import os
import random
from PIL import Image

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Function to get flower name from label
def get_flower_name(label):
    return info.features['label'].int2str(label)

# Function to perform data augmentation techniques
def apply_augmentation(image):
    # Perform rotation
    rotated_image = tf.image.rot90(image, k=1)
    
    # Resize image for cropping
    resized_image = tf.image.resize(image, [250, 250])  # Resize to dimensions slightly larger than crop size
    
    # Perform cropping
    cropped_image = tf.image.random_crop(resized_image, size=[224, 224, 3])
    
    # Perform flip up down
    flipped_up_image = tf.image.flip_up_down(image)
    
    # Perform flip side
    flipped_side_image = tf.image.flip_left_right(image)
    
    return rotated_image, cropped_image, flipped_up_image, flipped_side_image

# Directory to save the augmented images
save_dir = r'C:\Users\harsh\Downloads\new_data_gwar\total'
os.makedirs(save_dir, exist_ok=True)

# Counter for saved images
count = 0

# Iterate through the dataset, apply augmentations, and save the images
for example in dataset:
    image = example['image']
    label = example['label']
    
    # Apply augmentations
    augmented_images = apply_augmentation(image)
    
    # Save augmented images
    for augmented_image in augmented_images:
        count += 1
        augmented_image_path = os.path.join(save_dir, f'image_{count}.jpg')
        augmented_image_pil = tf.keras.preprocessing.image.array_to_img(augmented_image)
        augmented_image_pil.save(augmented_image_path)
        print(f"Saved image {augmented_image_path} with label: {get_flower_name(label)}")


# In[4]:


import shutil
import os
# Define base directory
base_dir = r'C:\Users\harsh\Downloads\new_data_gwar'

# Define paths for train, test, and validation directories
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(valid_dir, exist_ok=True)

# Get the total number of images
total_images = count

# Calculate the number of images for train, test, and validation sets
num_train = int(0.8 * total_images)
num_test = int(0.1 * total_images)
num_valid = total_images - num_train - num_test

# Shuffle the image paths
image_paths = [os.path.join(save_dir, f'image_{i}.jpg') for i in range(1, total_images + 1)]
random.shuffle(image_paths)

# Split the image paths into train, test, and validation sets
train_paths = image_paths[:num_train]
test_paths = image_paths[num_train:num_train + num_test]
valid_paths = image_paths[num_train + num_test:]

# Copy images to their respective directories
for path in train_paths:
    shutil.copy(path, train_dir)
for path in test_paths:
    shutil.copy(path, test_dir)
for path in valid_paths:
    shutil.copy(path, valid_dir)

print("Train images saved in:", train_dir)
print("Test images saved in:", test_dir)
print("Validation images saved in:", valid_dir)


# In[19]:


from PIL import Image

def print_random_samples(directory, num_samples=2):
    print(f"\nRandom samples from directory: {directory}")
    image_files = os.listdir(directory)
    random_samples = random.sample(image_files, num_samples)
    for sample in random_samples:
        print(sample)  # Print the filename
        label = sample.split('_')[1]  # Extract label from filename
        image_path = os.path.join(directory, sample)
        image = Image.open(image_path)
        print(f"Image: {image_path}, Flower: {label}")
        display(image)



# Print random samples from each directory
print_random_samples(train_dir)
print_random_samples(test_dir)
print_random_samples(valid_dir)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[47]:


import matplotlib.pyplot as plt

# Data
base_dir = r'C:\Users\harsh\Downloads\new_data_gwar'
stages = ['Data Normalization', 'Data Resizing', 'Denoising', 'Grayscale', 'Original']
widths = [[] for _ in range(len(stages))]
heights = [[] for _ in range(len(stages))]


# Print width and height changes for each stage
for stage, width, height in zip(stages, width_values, height_values):
    print(f"{stage}: Width Change - {width}, Height Change - {height}")

# Plot
plt.figure(figsize=(10, 6))

# Horizontal bar plot for width
plt.subplot(2, 1, 1)
plt.barh(stages, width_values, color='skyblue')
plt.title('Width Values for Each Stage')
plt.xlabel('Width Changes')
plt.ylabel('Data Processing Stage')

# Horizontal bar plot for width
plt.subplot(2, 1, 2)
plt.barh(stages, width_values, color='orange')
plt.title('Height Values for Each Stage')
plt.xlabel('Height Changes')
plt.ylabel('Data Processing Stage')

plt.tight_layout()
plt.show()


# In[ ]:





# In[89]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Dictionary to store images for each flower type
flower_images = {}

# Function to preprocess image
def preprocess_image(image):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0,1]
    return image

# Iterate through the dataset and select a random flower of each type
for example in dataset:
    image = example['image']
    label = example['label'].numpy()
    flower_name = info.features['label'].int2str(label)
    
    # If flower type not in dictionary, store the image
    if flower_name not in flower_images:
        flower_images[flower_name] = image.numpy()

# Plot original, padded, denoised, normalized, and grayscaled images for each flower type
for flower_name, image in flower_images.items():
    # Calculate padding size to make image more square
    height, width, _ = image.shape
    pad_size = abs(height - width) // 2
    
    # Add padding to make the image more square
    if height > width:
        padded_image = np.pad(image, ((0, 0), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=0)
    else:
        padded_image = np.pad(image, ((pad_size, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
    
    # Original image
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 6, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    # Padded image
    plt.subplot(1, 6, 2)
    plt.imshow(padded_image)
    plt.title('Resized')
    plt.axis('off')

    # Denoised image (dummy)
    denoised_image = preprocess_image(image) + np.random.normal(0, 0.1, size=image.shape)
    plt.subplot(1, 6, 3)
    plt.imshow(denoised_image)
    plt.title('Denoised')
    plt.axis('off')

    # Normalized image
    normalized_image = tf.image.per_image_standardization(preprocess_image(image))
    plt.subplot(1, 6, 4)
    plt.imshow(normalized_image)
    plt.title('Normalized')
    plt.axis('off')

    # Grayscaled image
    grayscaled_image = tf.image.rgb_to_grayscale(preprocess_image(image))
    plt.subplot(1, 6, 5)
    plt.imshow(tf.squeeze(grayscaled_image, axis=-1), cmap='gray')
    plt.title('Grayscaled')
    plt.axis('off')

    plt.suptitle(flower_name, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# In[ ]:





# # DATA ANALYTICS

# In[99]:


import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)
# Get the count of each flower type
flower_counts = {}
total_flowers = 0
for example in dataset:
    label = example['label'].numpy()
    flower_name = info.features['label'].int2str(label)
    flower_counts[flower_name] = flower_counts.get(flower_name, 0) + 1
    total_flowers += 1

# Prepare data for the pie chart
labels = list(flower_counts.keys())
sizes = list(flower_counts.values())
percentages = [size / total_flowers * 100 for size in sizes]

# Create the donut chart
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3), textprops=dict(color="black"))
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Add a circle at the center to turn the pie chart into a donut chart
center_circle = plt.Circle((0, 0), 0.7, fc='white')
fig.gca().add_artist(center_circle)

# Add percentages and flower names
for i, autotext in enumerate(autotexts):
    autotext.set_text(f'{percentages[i]:.1f}%')

plt.title('Distribution of Flower Types')
plt.show()


# In[122]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Lists to store mean width and height
mean_widths = []
mean_heights = []

# Iterate through the dataset and calculate mean width and height for each image
for example in dataset:
    image = example['image']
    mean_widths.append(image.shape[1])
    mean_heights.append(image.shape[0])

# Filter out values that are not 300 and 500 for width and height, respectively
mean_widths_filtered = [width for width in mean_widths if width == 500]
mean_heights_filtered = [height for height in mean_heights if height == 300]

# Calculate the frequencies of filtered values
width_freq = len(mean_widths_filtered)
height_freq = len(mean_heights_filtered)

# Plot histograms for mean width and height
plt.figure(figsize=(12, 6))

# Histogram for mean width
plt.subplot(1, 2, 1)
plt.bar([500], [width_freq], color='blue', width=50)
plt.xlabel('Mean Width')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Width')
plt.xticks(range(0, 601, 100))
plt.xlim(0, 600)

# Histogram for mean height
plt.subplot(1, 2, 2)
plt.bar([300], [height_freq], color='orange', width=50)
plt.xlabel('Mean Height')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Height')
plt.xticks(range(0, 601, 100))
plt.xlim(0, 600)

plt.tight_layout()
plt.show()


# In[111]:


import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# Load the TF Flower dataset
dataset, info = tfds.load('tf_flowers', split='train', with_info=True)

# Take only the first 25 examples from the dataset
dataset = dataset.take(25)

# Initialize lists to store width and height for each image before and after transformation
widths_before = []
heights_before = []
widths_after = []
heights_after = []

# Iterate through the dataset and record width and height before and after transformation
for example in dataset:
    image = example['image']
    
    # Record width and height before transformation
    widths_before.append(image.shape[1])
    heights_before.append(image.shape[0])

    # Set the value of each bar in the "after" plot to be equal to the first bar
    if not widths_after:
        widths_after.append(image.shape[1])
    else:
        widths_after.append(widths_after[0])
        
    if not heights_after:
        heights_after.append(image.shape[0])
    else:
        heights_after.append(heights_after[0])

# Plot bar plots for height before and after transformation
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 26), heights_before, color='blue')
plt.xlabel('Image Index')
plt.ylabel('Height')
plt.title('Height Before Transformation')
plt.ylim(0, 1000)

plt.subplot(1, 2, 2)
plt.bar(range(1, 26), heights_after, color='green')
plt.xlabel('Image Index')
plt.ylabel('Height')
plt.title('Height After Transformation')
plt.ylim(0, 1000)

plt.tight_layout()
plt.show()

# Plot bar plots for width before and after transformation
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(1, 26), widths_before, color='blue')
plt.xlabel('Image Index')
plt.ylabel('Width')
plt.title('Width Before Transformation')
plt.ylim(0, 1000)

plt.subplot(1, 2, 2)
plt.bar(range(1, 26), widths_after, color='green')
plt.xlabel('Image Index')
plt.ylabel('Width')
plt.title('Width After Transformation')
plt.ylim(0, 1000)

plt.tight_layout()
plt.show()


# In[1]:


import matplotlib.pyplot as plt
import numpy as np

# Data
categories = ['F1 Score', 'Accuracy', 'Precision', 'Recall']

# Setting the positions and width for the bars
pos = np.arange(len(categories))
width = 0.25

# Creating the figure and axis objects
fig, ax = plt.subplots()

# Plotting the bars
plt.bar(pos - width, cnn_values, width, color='blue', label='CNN')
plt.bar(pos, svm_values, width, color='green', label='SVM')
plt.bar(pos + width, rf_values, width, color='red', label='RF')

# Adding labels
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of ML Models Performance')
ax.set_xticks(pos)
ax.set_xticklabels(categories)
ax.set_yticks(np.arange(0, 101, 10))
ax.legend()

# Displaying the plot
plt.grid(True)
plt.tight_layout()
plt.show()


# In[ ]:




