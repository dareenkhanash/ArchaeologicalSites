
# Image Classification for Archaeological Sites using Pre-trained Models

### Overview

Archaeological research often involves the analysis of images to identify and classify various archaeological sites. In this project, we are developing an advanced deep learning model that automatically classifies images of archaeological sites, with the utilization of pre-trained models that plays a crucial role in achieving accurate and efficient results.

---

### Dependencies and Setup

- **Python Libraries:** Extensive use of libraries such as NumPy, Pandas, TensorFlow Keras, Scikit-learn, and PIL for data manipulation, model building, and image processing.
- **Pre-trained Models:** Use of models like InceptionV3, EfficientNetB0, ResNet50, and VGG16 from TensorFlow Keras.
- **Data Handling:** Methods for loading, processing, and augmenting image data for neural network training.
---

### Data Collection
- The primary source for gathering image data was the TripAdvisor website.
- Focused on six distinct archaeological sites: Umm Qais, Jerash, Petra, Ajloun Castle, Wadi Rum, and the Roman Amphitheater.
- Approximately 500 to 800 images were systematically collected for each archaeological site class.

The dataset used in this project is located within the designated "data" folder.

---

### Preprocessing Steps
1. **Image Loading and Conversion:** Detailed methods for loading images and converting them into a suitable format for neural network input.
2. **Data Augmentation:** Application of various techniques to enhance the dataset and prevent overfitting.
3. **Model-Specific Preprocessing:** Use of specific preprocessing functions for each model to ensure compatibility with input data.

---

### Models Description
- **InceptionV3:** Efficient for classifying images into numerous categories.
- **EfficientNetB0:** Balances model scaling in depth, width, and resolution.
- **ResNet50:** Utilizes residual connections for training deeper networks.
- **VGG16:** Employs small convolutional filters in deep architectures.

---

### Results and Visualizations

| Model                 | Accuracy | Precision | Recall   |
|-----------------------|----------|-----------|----------|
| InceptionV3           | 0.86     | 0.86      | 0.86     |
| EfficientNetB0        | 0.83     | 0.85      | 0.83     |
| ResNet50              | 0.80     | 0.81      | 0.82     |
| VGG16                 | 0.75     | 0.76      | 0.75     |



# Live App 
To experience the live app, please visit: [Live App Link](https://huggingface.co/spaces/DareenY/archaeological_sites).

---

### Code Explanation
- **Model Customization:** Details on how each model is loaded and customized with additional layers for specific tasks.
- **Training Process:** Description of the training process, including hyperparameters and optimization techniques.
- **Performance Metrics:** Explanation of the metrics used to evaluate model effectiveness.

---
