
# Image Classification for Archaeological Sites using Pre-trained Models

### Overview

Archaeological research often involves the analysis of images to identify and classify various archaeological sites. In this project, we aim to develop a deep learning model for automated image classification of archaeological sites. The utilization of pre-trained models will play a crucial role in achieving accurate and efficient results.

---

### Objectives
- Develop a deep learning model for image classification
- Evaluate multiple pre-trained models: InceptionV3, Efficientnet, ResNet, VGGNet
- Compare and analyze the performance of each model
- Provide a user-friendly demo using Gradio or streamlit
---

### Dependencies and Setup
- **Python Libraries:** Extensive use of libraries such as NumPy, Pandas, TensorFlow Keras, Scikit-learn, and PIL for data manipulation, model building, and image processing.
- **Pre-trained Models:** Use of models like InceptionV3, EfficientNetB0, ResNet50, and VGG16 from TensorFlow Keras.
- **Data Handling:** Methods for loading, processing, and augmenting image data for neural network training.
---

### Data Collection
- Utilized the TripAdvisor website as the primary source for gathering image data.
- Focused on six distinct archaeological sites: Umm Qais, Jerash, Petra, Ajloun Castle, Wadi Rum, and the Roman Amphitheater.
- Systematically collected approximately 500 to 800 images for each archaeological site class.
    
---

### Preprocessing Steps
1. **Image Loading and Conversion:** Detailed methods for loading images and converting them into a suitable format for neural network input.
2. **Data Augmentation:** Application of various techniques to enhance the dataset and prevent overfitting.
3. **Model-Specific Preprocessing:** Use of specific preprocessing functions for each model to ensure compatibility with input data.

---

### Model Descriptions
- **InceptionV3:** Efficient for classifying images into numerous categories.
- **EfficientNetB0:** Balances model scaling in depth, width, and resolution.
- **ResNet50:** Utilizes residual connections for training deeper networks.
- **VGG16:** Employs small convolutional filters in deep architectures.

---

### Code Explanation
- **Model Customization:** Details on how each model is loaded and customized with additional layers for specific tasks.
- **Training Process:** Description of the training process, including hyperparameters and optimization techniques.
- **Performance Metrics:** Explanation of the metrics used to evaluate model effectiveness.

---

### Results and Visualizations

| Model                 | Accuracy | Precision | Recall   |
|-----------------------|----------|-----------|----------|
| InceptionV3           | 0.86     | 0.86      | 0.86     |
| Efficientnet          | 0.83     | 0.85      | 0.83     |
| ResNet                | 0.80     | 0.81      | 0.82     |
| VGG                   | 0.75     | 0.76      | 0.75     |



# Live App 
To experience the live app, please visit:

https://huggingface.co/spaces/DareenY/archaeological_sites

---



### Complete Technical Documentation
---

# Requirements 

To download all requirements run:
```python
pip install -r requirements.txt

```


# Function to load and preprocess images
def load_and_preprocess_data(output_folder):
    data_inception = []
    data_efficientnet = []
    data_resnet = []
    data_vgg = []
    labels = []

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for label, folder in enumerate(os.listdir(main_folder)):
        folder_path = os.path.join(main_folder, folder)
        
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)

            # Load and preprocess image for InceptionV3
            img_inception = load_img(img_path, target_size=(299, 299))
            img_array_inception = img_to_array(img_inception)
            
            # Apply data augmentation
            img_array_inception = datagen.random_transform(img_array_inception)
            
            img_array_inception = preprocess_inception(img_array_inception)
            data_inception.append(img_array_inception)

            # Load and preprocess image for EfficientNet
            img_efficientnet = load_img(img_path, target_size=(224, 224))
            img_array_efficientnet = img_to_array(img_efficientnet)
            
            # Apply data augmentation
            img_array_efficientnet = datagen.random_transform(img_array_efficientnet)
            
            img_array_efficientnet = preprocess_efficientnet(img_array_efficientnet)
            data_efficientnet.append(img_array_efficientnet)

            # Load and preprocess image for ResNet
            img_resnet = load_img(img_path, target_size=(224, 224))
            img_array_resnet = img_to_array(img_resnet)
            
            # Apply data augmentation
            img_array_resnet = datagen.random_transform(img_array_resnet)
            
            img_array_resnet = preprocess_resnet(img_array_resnet)
            data_resnet.append(img_array_resnet)

            # Load and preprocess image for VGGNet
            img_vgg = load_img(img_path, target_size=(224, 224))
            img_array_vgg = img_to_array(img_vgg)
            
            # Apply data augmentation
            img_array_vgg = datagen.random_transform(img_array_vgg)
            
            img_array_vgg = preprocess_vgg(img_array_vgg)
            data_vgg.append(img_array_vgg)

            labels.append(label)

    return {
        "inception": np.array(data_inception),
        "efficientnet": np.array(data_efficientnet),
        "resnet": np.array(data_resnet),
        "vgg": np.array(data_vgg),
    }, np.array(labels)

# Call the function
data, labels = load_and_preprocess_data(main_folder)

```python
label_to_site = {labels: site for labels, site in enumerate(os.listdir(main_folder))}
np.unique(labels)
labels_encoded = to_categorical(labels)
```

```python
label_to_site
```

# InceptionV3 Model

#InceptionV3_train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['inception'], labels_encoded, test_size=0.2, random_state=42)

# Load the InceptionV3 model with pre-trained weights on ImageNet
InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze the layers of the InceptionV3 base model
for layer in InceptionV3_base_model.layers:
    layer.trainable = False

```python
x = InceptionV3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(600, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)

predictions = Dense(len(label_to_site), activation='softmax')(x)

# Create the fine-tuned model
InceptionV3_fine_tuned_model = Model(inputs=InceptionV3_base_model.input, outputs=predictions)

# Compile the model with the correct learning rate argument
InceptionV3_fine_tuned_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create the early stopping callback
early_stopping_InceptionV3 = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
```

# Train the model with early stopping
InceptionV3_fine_tuned_model.fit(X_train, y_train, 
                     epochs=15, batch_size=32, 
                     validation_data=(X_test, y_test), 
                     callbacks=[early_stopping_InceptionV3])

```python
InceptionV3_losses = pd.DataFrame(InceptionV3_fine_tuned_model.history.history)
InceptionV3_losses
```

```python
InceptionV3_losses.plot()
```

```python
InceptionV3_pred = InceptionV3_fine_tuned_model.predict(X_test)
```

```python
y_test[0] , InceptionV3_pred[0]
```

```python
InceptionV3_post =np.where(InceptionV3_pred >= 0.5, 1,0)
```

```python
InceptionV3_pred[0], InceptionV3_post[0], y_test[0]
```

```python
InceptionV3_pred.min(),InceptionV3_pred.max(), InceptionV3_pred.mean()
```

#classification_report for Inception model
print(classification_report(y_test, InceptionV3_post))

#Test some images using InceptionV3 prediction
test_InceptionV3_image = "/kaggle/input/finaltest/3.jpeg"

# Load and preprocess the image
img = image.load_img(test_InceptionV3_image, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_inception(img_array)

# Make a prediction
predictions = InceptionV3_fine_tuned_model.predict(img_array)

num_classes = 6
predicted_class_index = np.argmax(predictions)
predicted_class_label = label_to_site[predicted_class_index]

print(f"Predicted Class: {predicted_class_label}")


# Save the weights to an H5 file
InceptionV3_fine_tuned_model.save('Final_Inception_model.h5')

# **EfficientNetB0 Model**

#train_test_split EfficientNetB0 data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['efficientnet'], labels_encoded, test_size=0.2, random_state=42)

# Load the EfficientNetB0 model with pre-trained weights on ImageNet
Efficientnet_base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in Efficientnet_base_model.layers:
    layer.trainable = False

#classification layers
x = Efficientnet_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(500, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)

predictions = Dense(len(label_to_site), activation='softmax')(x)

# Create the fine-tuned model
fine_tuned_efficientnet_model = Model(inputs=Efficientnet_base_model.input, outputs=predictions)

# Compile the model
fine_tuned_efficientnet_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create the early stopping callback
Efficientnet_early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
fine_tuned_efficientnet_model.fit(X_train, y_train, 
                     epochs=10, batch_size=32, 
                     validation_data=(X_test, y_test), 
                     callbacks=[Efficientnet_early_stopping])

```python
losses_efficientnet_model = pd.DataFrame(fine_tuned_efficientnet_model.history.history)
losses_efficientnet_model
```

```python
losses_efficientnet_model.plot()
```

```python
efficientnet_pred = fine_tuned_efficientnet_model.predict(X_test)
```

```python
efficientnet_post =np.where(efficientnet_pred >= 0.5, 1,0)
```

#classification_report for efficientnet model
print(classification_report(y_test, efficientnet_post))

#Test some images using InceptionV3 model
test_EfficientNetB0_image = "/kaggle/input/finaltest/3.jpeg"

# Load and preprocess the image
img = image.load_img(test_EfficientNetB0_image, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_efficientnet(img_array)

# Make a prediction
predictions = fine_tuned_efficientnet_model.predict(img_array)

num_classes = 6
predicted_class_index = np.argmax(predictions)
predicted_class_label = label_to_site[predicted_class_index]

print(f"Predicted Class: {predicted_class_label}")


# Save to H5 file
fine_tuned_efficientnet_model.save('Final_efficientnet_model.h5')

# ResNet50 Model

#ResNet50_Model_train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['resnet'], labels_encoded, test_size=0.2, random_state=42)

# Load the ResNet50 model with pre-trained weights on ImageNet
resnet_base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in resnet_base_model.layers:
    layer.trainable = False

```python
x = resnet_base_model.output
x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
#x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Flatten()(x) 
x = Dense(556, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x) 
x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
#x = BatchNormalization()(x)
x = Dropout(0.5)(x)

resnet_predictions = Dense(len(label_to_site), activation='softmax')(x)

# Create the fine-tuned model
fine_tuned_resnet_model = Model(inputs=resnet_base_model.input, outputs=resnet_predictions)

# compile the model
fine_tuned_resnet_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Create the early stopping callback
resnet_early_stopping = EarlyStopping(monitor='val_loss', patience=1,verbose=1,restore_best_weights=True)
```

# Train the model with early stopping
fine_tuned_resnet_model.fit(X_train, y_train, 
                     epochs=20, batch_size=32, 
                     validation_data=(X_test, y_test), 
                     callbacks=[resnet_early_stopping])

```python
resnet_losses_model = pd.DataFrame(fine_tuned_resnet_model.history.history)
resnet_losses_model
```

```python
resnet_losses_model.plot()
```

```python
resnet_pred = fine_tuned_resnet_model.predict(X_test)
```

```python
resnet_post =np.where(resnet_pred >= 0.5, 1,0)
```

#classification_report for resnet model
print(classification_report(y_test, resnet_post))

```python
fine_tuned_resnet_model.save('Final_RESNET_model.h5')
```

#Test some images using RESNET model
test_resnet_image = "/kaggle/input/finaltest/3.jpeg"

# Load and preprocess the image
img = image.load_img(test_resnet_image, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_resnet(img_array)

# Make a prediction
predictions = fine_tuned_resnet_model.predict(img_array)

num_classes = 6
predicted_class_index = np.argmax(predictions)
predicted_class_label = label_to_site[predicted_class_index]

print(f"Predicted Class: {predicted_class_label}")


# VGG16 Model

#vgg_Model_train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['vgg'], labels_encoded, test_size=0.2, random_state=42)

# Load the vgg model with pre-trained weights on ImageNet
vgg_base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
for layer in vgg_base_model.layers:
    layer.trainable = False

```python
x = vgg_base_model.output
x = Conv2D(filters=512, kernel_size=(3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
x = Flatten()(x) 
x = Dense(556, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
x = Dropout(0.5)(x) 

# Output layer with softmax activation for multiclass categorical task
vgg_predictions = Dense(len(label_to_site), activation='softmax')(x)

# Create the fine-tuned VGG model
fine_tuned_vgg_model = Model(inputs=vgg_base_model.input, outputs=vgg_predictions)

# Compile the model
fine_tuned_vgg_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#fine_tuned_vgg_model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Create the early stopping callback
vgg_early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
```

# Train the model with early stopping
fine_tuned_vgg_model.fit(X_train, y_train, 
                         epochs=30, batch_size=32, 
                         validation_data=(X_test, y_test), 
                         callbacks=[vgg_early_stopping])

```python
vgg_losses_model = pd.DataFrame(fine_tuned_vgg_model.history.history)
vgg_losses_model
```

```python
vgg_losses_model.plot()
```

```python
vgg_pred = fine_tuned_vgg_model.predict(X_test)
vgg_post =np.where(vgg_pred >= 0.5, 1,0)
```

#classification_report for resnet model
print(classification_report(y_test, vgg_post))

```python
test_vgg_image = "//kaggle/input/testtt/5.jpg"

# Load and preprocess the image
img = image.load_img(test_vgg_image, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_vgg(img_array)

# Make a prediction
predictions = fine_tuned_vgg_model.predict(img_array)

num_classes = 6
predicted_class_index = np.argmax(predictions)
predicted_class_label = label_to_site[predicted_class_index]

print(f"Predicted Class: {predicted_class_label}")

```

```python
fine_tuned_vgg_model.save('Final_VGG_model.h5')
```

```python

```
