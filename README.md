# 🤖 Face Mask Detection Using CNN  

## Project Overview  
This project focuses on building a **Convolutional Neural Network (CNN)** model to automatically detect whether a person is wearing a **face mask** or **not** using image data.  
The model was trained on a labeled dataset of images containing masked and unmasked faces, achieving efficient performance in real-time detection — useful for public safety and health monitoring applications, especially during pandemic situations.

---

## Objective  
The main objective of this project is to:
- Develop a **deep learning model** capable of classifying face images into two categories:
  - **With Mask**
  - **Without Mask**
- Enhance public safety by enabling **automated mask detection** through computer vision.
- Deploy the trained model for use in real-time applications such as **CCTV monitoring systems**, **attendance systems**, or **entry gate scanning**.

---

## Dataset  
- **Dataset Name:** Face Mask Detection Dataset  
- **Source:** [Kaggle - Face Mask Dataset by Omkar Gurav](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  
- The dataset consists of two folders:
  - `WithMask`
  - `WithoutMask`  
- Images are well-balanced and suitable for binary classification.

---

## Tools & Technologies Used  
- **Programming Language:** Python  
- **Environment:** Google Colab  
- **Version Control:** GitHub  
- **Dataset Source:** Kaggle  

---

## Libraries Used  

| Library | Purpose |
|----------|----------|
| `numpy` | Numerical computation |
| `pandas` | Data manipulation |
| `matplotlib` / `seaborn` | Data visualization |
| `tensorflow` / `keras` | Building and training CNN models |
| `opencv` | Image preprocessing and resizing |
| `sklearn` | Data splitting and performance metrics |
| `zipfile` | Extracting dataset from compressed format |

---

## Methodology  

1. **Data Collection:**  
   Downloaded the dataset from Kaggle.  

2. **Data Preprocessing:**  
   - Resized all images to uniform dimensions (e.g., 128x128).  
   - Normalized pixel values for better model convergence.  
   - Applied data augmentation to prevent overfitting.  

3. **Model Building:**  
   - Used **Convolutional Neural Network (CNN)** architecture.  
   - Layers included Conv2D, MaxPooling, Flatten, Dense, and Dropout.  

4. **Model Training:**  
   - Split data into training and validation sets.  
   - Used `Adam` optimizer and `binary_crossentropy` loss function.  

5. **Model Evaluation:**  
   - Evaluated using accuracy, loss curves, and confusion matrix.  

6. **Prediction:**  
   - Model predicts whether a person is wearing a mask or not based on input image.  

---

## Results  
- **Training Accuracy:** ~98%  
- **Validation Accuracy:** ~96%  
- The model successfully identifies masked and unmasked faces with high precision.  
- Performance was visualized using accuracy and loss graphs.

---

## Conclusion  
- The CNN model effectively distinguishes between people **with** and **without masks**.  
- Can be deployed in real-world scenarios like:
  - Public monitoring systems
  - Office entry gates
  - Healthcare facilities  
- Future improvements can include:
  - Using larger and more diverse datasets.
  - Implementing real-time detection using OpenCV and webcam input.
  - Deploying the model as a **web or mobile app**.

---

## Future Scope  
- Integration with **IoT devices** for automated monitoring.  
- **Mobile deployment** using TensorFlow Lite.  
- Real-time video stream detection.
