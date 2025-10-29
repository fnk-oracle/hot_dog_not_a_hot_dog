# 🍔 Hotdog vs Not Hotdog Classifier

A TensorFlow-based binary image classifier trained on the **[Food-101 dataset](https://www.tensorflow.org/datasets/catalog/food101)** to distinguish between *hotdogs* and *non-hotdogs* — inspired by the "Silicon Valley" app!

---

## 🧠 Project Overview

This project uses the Food-101 dataset to build a **deep learning image classifier** that determines whether a given food image is a **hotdog (class 55)** or **not a hotdog**.

The model is implemented using:

* **TensorFlow 2 / Keras**
* **TensorFlow Datasets (TFDS)**
* **Convolutional Neural Networks (CNNs)**
* **Data Augmentation** and **Regularization** for generalization

---

## ⚙️ Features

* ✅ Uses Food-101’s 100,000+ labeled food images
* ✅ Resizes all images to 128×128 for faster training
* ✅ Balances hotdog vs non-hotdog samples
* ✅ Includes real-time data augmentation (`RandomFlip`, `RandomRotation`)
* ✅ Applies dropout and L2 regularization
* ✅ Tracks accuracy and validation loss across 50 epochs

---

## 📚 Dataset

The project uses the **Food-101** dataset available via `tensorflow_datasets`.

Each image belongs to one of 101 food categories.
Hotdogs correspond to **label index 55**.

Dataset is automatically downloaded using:

```python
ds, ds_info = tfds.load('food101', as_supervised=True, with_info=True)
```

---

## 🧩 Model Architecture

| Layer               | Description                            |
| ------------------- | -------------------------------------- |
| Input (128×128×3)   | Normalized RGB image                   |
| Data Augmentation   | Random horizontal flips & rotations    |
| Conv2D (64 filters) | ReLU activation + MaxPooling + Dropout |
| Conv2D (64 filters) | ReLU + L2 regularization               |
| Conv2D (32 filters) | ReLU + L2 regularization               |
| Flatten             | Converts features into a vector        |
| Dense (128)         | ReLU + Dropout                         |
| Dense (1)           | Sigmoid output (binary classification) |

**Optimizer:** Adam (learning rate = 1e-3)
**Loss:** Binary Crossentropy
**Metric:** Accuracy

---

## 🧪 Training

Run the notebook or Python script:

```python
history = model.fit(train_ds, validation_data=valid_ds, epochs=50)
```

Typical output:

```
Epoch 1/50
...
Train accuracy: 0.91
Validation accuracy: 0.89
```

---

## 📈 Results Visualization

Plot training and validation performance:

```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training vs Validation Accuracy')
plt.show()
```

---

## 🔍 Testing the Model

Predict whether a new image is a hotdog:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('test_image.jpg', target_size=(128, 128))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
prediction = model.predict(img_array)[0][0]

if prediction > 0.5:
    print("🌭 Hotdog detected!")
else:
    print("❌ Not a hotdog!")
```

---

## 🧰 Troubleshooting

**Error:** `expected shape=(None, 512, 512, 3), found shape=(None, 128, 128, 3)`
→ Restart runtime and rebuild model with `tf.keras.backend.clear_session()` after changing `MAX_SIDE_LEN`.

**Error:** `expected axis -1 to have value 492032, got 25088`
→ Model was compiled for old input size. Restart kernel and rebuild after dataset preprocessing.

---

## 🏁 Future Improvements

* Use transfer learning (e.g., MobileNetV2 or EfficientNet) for faster convergence
* Implement a web app with TensorFlow.js or Flask for live classification
* Add more data augmentations and fine-tuning options

---

## 👨‍💻 Author

**Raunak Srivastava**

Computer Vision, Deep Learning, AI Enthusiast*
