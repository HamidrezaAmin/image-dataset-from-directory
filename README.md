# Image Classification with TensorFlow `image_dataset_from_directory`

This project demonstrates how to load and preprocess image data using `tf.keras.utils.image_dataset_from_directory`, build a Convolutional Neural Network (CNN) using TensorFlow/Keras, and train a model to classify flower images into five categories.

## 📁 Dataset

The dataset consists of flower images downloaded from TensorFlow's public dataset link:

```bash
!wget https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
!tar -xvzf flower_photos.tgz
```

The data is automatically structured into subdirectories, where each folder name is a class label (e.g., `daisy`, `dandelion`, `roses`, `sunflowers`, `tulips`).

## 🔧 Setup

Install TensorFlow (latest version recommended):

```python
!pip install tensorflow --upgrade
```

Import the required libraries:

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
```

## 📥 Load and Prepare Dataset

Load training and validation datasets from directory with 80/20 split:

```python
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)
```

## 📊 Standardize the Data

Normalize pixel values from [0, 255] to [0, 1]:

```python
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
```

## ⚙️ Configure for Performance

```python
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

## 🧠 Model Architecture

```python
num_classes = 5

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),  # Skip if using normalized_ds
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
])
```

## 🧪 Compile and Train

```python
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

history = model.fit(
    train_ds,  # or normalized_ds
    validation_data=val_ds,
    epochs=30
)
```

## 📈 Evaluation and Results

During training, the model accuracy and loss are tracked per epoch. You can also visualize training history using Matplotlib.

## 🔗 Reference

- TensorFlow Image Classification: [Image Classification](https://www.tensorflow.org/tutorials/images/classification)
- Performance Tips: [tf.data API Guide](https://www.tensorflow.org/guide/data_performance)
