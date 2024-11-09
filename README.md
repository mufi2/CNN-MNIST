# 🧠 MNIST Digit Recognition using CNN

An advanced **Convolutional Neural Network (CNN)** model for accurately recognizing handwritten digits from the MNIST dataset. This project combines a powerful model training pipeline and an interactive web application built with Streamlit for real-time predictions.

---

## 📖 Introduction
This project leverages CNNs to classify handwritten digits, offering a precise solution for the MNIST classification challenge. Using this model, images of digits from 0-9 are recognized with high accuracy, making it suitable for educational purposes, exploratory data science, and model deployment.

---

## 🧬 Model Architecture
The CNN model architecture comprises:
- **Two Convolutional Layers** with ReLU activations, each followed by Max Pooling.
- **Fully Connected Layer** for final classification, mapping features to digit classes.
- **Softmax Activation** to produce a probability distribution across the 10 digit classes.

![image](https://github.com/user-attachments/assets/c6153aa6-d145-429b-a152-64fac30ff25b)

---

## 📈 Training Process
### Hyperparameters
- **Learning Rate**: 0.0003
- **Batch Size**: 64
- **Epochs**: 15

During training, the model's loss and accuracy are evaluated at each epoch. This monitoring helps ensure that the model generalizes well on the test data.

### 🔹 Training Loss
The plot below demonstrates the loss reduction across epochs, showing the model's learning progression:
![Training Loss Plot](images/training_loss.png)

### 🔹 Accuracy
- **Training Accuracy**: 98.30%
- **Test Accuracy**: 97.50%

---

## 🚀 Streamlit Web Application
An interactive [Streamlit web application]([ADD_YOUR_STREAMLIT_APP_LINK_HERE](https://cnn-mnist-uvtrszxczn3qcd3yt4m6am.streamlit.app/)) enables users to upload images of handwritten digits and get predictions in real time, powered by the trained CNN model.

### 💻 App Interface
The app features a sleek and intuitive interface:
![Streamlit App Screenshot](![image](https://github.com/user-attachments/assets/61860ceb-951a-42df-9ff0-2e601f92db14))

Upload an image, and the model will instantly classify the digit with high precision.

---

## 🔑 Key Features
- **High Accuracy**: Optimized CNN model achieves significant accuracy on the MNIST dataset.
- **User-Friendly Web App**: The Streamlit interface provides seamless, real-time interaction for predictions.
- **Efficient Model**: Optimized for fast inference, allowing swift digit classification on uploaded images.

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).
