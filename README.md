# ✍️ Handwritten Digit Recognition using Machine Learning

A **Machine Learning & Deep Learning project** that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**.  
This project demonstrates the complete ML workflow including data preprocessing, model training, evaluation, and prediction on custom handwritten images.

---

## 📌 Project Overview

Handwritten Digit Recognition is a fundamental problem in **computer vision and pattern recognition**.  
This project builds a CNN-based model capable of accurately classifying handwritten digits despite variations in writing styles, size, and stroke patterns.

The model is trained on the **MNIST dataset**, consisting of 70,000 grayscale images, and achieves **high accuracy (>98%)** on test data.

---

## 🎯 Objectives

- To design and implement a **CNN model** for handwritten digit classification  
- To preprocess and normalize image data for optimal performance  
- To train, validate, and evaluate the model using standard metrics  
- To test the model on **custom handwritten digit images**  
- To gain hands-on experience with **deep learning and computer vision**

---

## 🧠 Dataset Used

**MNIST Dataset**
- 70,000 grayscale images of digits (0–9)
- Image size: **28 × 28 pixels**
- Training set: 60,000 images
- Test set: 10,000 images

---

## 🛠️ Technologies & Tools

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Matplotlib**
- **OpenCV (cv2)**
- **Jupyter Notebook / Python Script**

---

## ⚙️ Model Architecture

The CNN model consists of:
- Convolutional layers with ReLU activation
- Max Pooling layer
- Dropout layers for regularization
- Fully connected dense layer
- Softmax output layer (10 classes)

**Loss Function:** Sparse Categorical Crossentropy  
**Optimizer:** Adam  
**Evaluation Metric:** Accuracy  

---

## 🔄 Workflow

1. Load MNIST dataset  
2. Normalize and reshape images  
3. Build CNN architecture  
4. Train model with validation split  
5. Evaluate model on test dataset  
6. Save trained model  
7. Predict digits from custom images  

---

## 📈 Results & Performance
- **Training Accuracy:** ~99%
- **Validation Accuracy:** ~98–99%
- **Test Accuracy:** >98%
- Strong performance on both MNIST and custom handwritten images

---

## 📂 Project Structure

Handwritten-Digit-Recognition/
│
├── handwritten_digit_recognition_model.py
├── hand_written_number_recognition_model.ipynb
├── digit_recogniser_model.h5
├── README.md
├── sample_images/
└── project_report.pdf
