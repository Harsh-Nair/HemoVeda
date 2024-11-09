import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import cv2

# Load the model
model = tf.keras.models.load_model('palm_resnet50.h5')

# Image preprocessing function
def preprocess_input(img_path):
    img = image.load_img(img_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

# Grad-CAM function
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute the gradient of the top predicted class for the image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Compute the gradient of the class output value with respect to the feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # Pool the gradients across the width and height dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 and 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Display Grad-CAM overlay
def overlay_gradcam(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))

    # Rescale heatmap to a range of 0-255 and resize it to match the image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img

    # Display Grad-CAM
    cv2.imshow('Grad-CAM', np.uint8(superimposed_img))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define the layer to visualize
last_conv_layer_name = 'conv5_block3_out'  # Adjust this layer based on your model

# Sample images for Grad-CAM
sample_images = {
    'Anemic': [
        r"D:\Hack Wars-Team Advait\Anemic-271.png"
    ],
    'Healthy': [
        r"D:\Hack Wars-Team Advait\non-anemic.jpg"
    ],
}

# Iterate through each class and its images
for class_name, images in sample_images.items():
    for image_path in images:
        img_array = preprocess_input(image_path)
        
        # Make prediction and generate heatmap
        preds = model.predict(img_array)
        pred_class = np.argmax(preds[0])  # 0: Anemic, 1: Healthy
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_class)
        
        # Display Grad-CAM overlay
        overlay_gradcam(image_path, heatmap)
        