#!/usr/bin/env python
# coding: utf-8

# In[4]:


import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk, Image
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


# In[5]:


# Load Flowers dataset from TensorFlow
flowers_data = tf.keras.utils.get_file(
    'flower_photos',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    untar=True
)

# Preprocessing
batch_size = 32
img_height = 224
img_width = 224

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  flowers_data,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  flowers_data,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[6]:


# CNN for feature extraction
data = r"C:\Users\harsh\Downloads\new_data_gwar\total\train"
cnn_model = tf.keras.applications.MobileNetV2(input_shape=(img_height, img_width, 3),
                                               include_top=False,
                                               weights='imagenet')

def extract_features(dataset):
    features = []
    labels = []
    for img_batch, label_batch in dataset:
        features_batch = cnn_model.predict(img_batch)
        features.extend(features_batch)
        labels.extend(label_batch.numpy())
    return np.array(features), np.array(labels)

X_train_cnn, y_train_cnn = extract_features(train_ds)
X_val_cnn, y_val_cnn = extract_features(val_ds)

# Ensemble of SVM and RF
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_cnn.reshape(X_train_cnn.shape[0], -1), y_train_cnn)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train_cnn.reshape(X_train_cnn.shape[0], -1), y_train_cnn)


# In[18]:


# Load flower details from CSV
flower_details = pd.read_csv('about.csv')

# UI using Tkinter
def classify_flower():
    global canvas, label
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path)
        img = img.resize((img_height, img_width), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)
        
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.image = img
        canvas.delete("details")
        
        img_array = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(file_path, target_size=(img_height, img_width)))
        img_array = tf.expand_dims(img_array, 0)
        features = cnn_model.predict(img_array)
        
        svm_prediction = svm_model.predict(features.reshape(1, -1))[0]
        rf_prediction = rf_model.predict(features.reshape(1, -1))[0]
        
        svm_flower = flower_details.loc[flower_details['Class'] == svm_prediction]  # Change 'Label' to the correct column name
        rf_flower = flower_details.loc[flower_details['Class'] == rf_prediction]  # Change 'Label' to the correct column name
        
        if svm_prediction == rf_prediction:
            messagebox.showinfo("Flower Classification", f"The flower is: {svm_flower['Flower'].values[0]}\n\nSVM Details:\n{svm_flower.to_string(index=False)}\n\nRF Details:\n{rf_flower.to_string(index=False)}")
        else:
            messagebox.showwarning("Ensemble Discrepancy", "SVM and Random Forest predictions don't match.")
        
        show_details(svm_flower)

def show_details(flower):
    top = tk.Toplevel()
    top.title("Flower Details")
    
    detail_label = tk.Label(top, text=flower.to_string(index=False))
    detail_label.pack()

root = tk.Tk()
root.title("Flower Classification")

canvas = tk.Canvas(root, width=img_height, height=img_width)
canvas.pack()

upload_button = tk.Button(root, text="Upload Image", command=classify_flower)
upload_button.pack()

submit_button = tk.Button(root, text="Submit", command=classify_flower)
submit_button.pack()

root.mainloop()




# In[17]:


import tkinter as tk
from tkinter import ttk
import pandas as pd

# Load flower details from CSV
flower_details = pd.read_csv('about.csv')

# Function to display flower details in tabular format
def display_flower_details(flower_index):
    # Clear previous details
    for widget in details_frame.winfo_children():
        widget.destroy()
    
    # Fetch flower details
    flower = flower_details.iloc[flower_index]
    
    # Display about_flower details in paragraph format
    about_flower_label = tk.Label(details_frame, text="About Flower:")
    about_flower_label.pack()
    
    about_flower_text = tk.Text(details_frame, wrap=tk.WORD, height=10, width=70)
    about_flower_text.insert(tk.END, flower['about_flower'])
    about_flower_text.pack(pady=10)
    
    # Display details in tabular format
    details_table = ttk.Treeview(details_frame, columns=["Attribute", "Details"], show="headings")
    details_table.heading("Attribute", text="Attribute")
    details_table.heading("Details", text="Value")
    
    for col, val in zip(flower.index[1:], flower.values[1:]):
        details_table.insert("", "end", values=(col, val))
    
    details_table.pack()

# Create main window
root = tk.Tk()
root.title("Flower Details")

# Frame to hold details
details_frame = tk.Frame(root)
details_frame.pack(pady=20)

# Fetch and display details for the 1st flower
display_flower_details(0)  # Indexing starts from 0

root.mainloop()


# In[ ]:




