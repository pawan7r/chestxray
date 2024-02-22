from keras_preprocessing import image
from keras.models import load_model, Model
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
from keras.layers import Dense, Flatten

# Load the original model
model = load_model('our_model.h5')

# Define VGG16 as a feature extractor
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of VGG16 to prevent them from being trained
for layer in base_model.layers:
    layer.trainable = False

# Extract features using VGG16
vgg16_features = base_model.output
vgg16_features = Flatten()(vgg16_features)

# Combine VGG16 features with the original model's output
combined_output = Dense(2, activation='softmax')(vgg16_features)  # Adjust the output shape based on your task

# Create a new model with the combined output
combined_model = Model(inputs=base_model.input, outputs=combined_output)

# Load and preprocess the image
img_path = 'D:\\search items\\chestxray\static\\val_dir\\PNEUMONIA\\oioio.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_data = preprocess_input(img_array)

# Make predictions using the combined model
prediction = combined_model.predict(img_data)

# Continue with your existing code for printing the result
if prediction[0][0] > prediction[0][1]:
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')

print(f'Predictions: {prediction}')