import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def make_gradcam_heatmap(
    img_path,
    model,
    last_conv_layer_name,
    classifier_layer_names,
    size=(224, 224)
):
 
    """
    Generates a Grad-CAM heatmap from an image and a model.

    Args:
        img_path (str): Path to the preprocessed image.
        model (tf.keras.Model): Loaded model.
        last_conv_layer_name (str): Name of the last convolutional layer.
        classifier_layer_names (List[str]): Names of the classifier (dense) layers.
        size (tuple): Size to which the heatmap is resized.

    Returns:
        np.ndarray: Heatmap of shape (height, width, 3), values in [0, 255].
    """
    #Load & prepare the image
    img = load_img(img_path, color_mode='grayscale', target_size=size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model that outputs the last conv layer's output
    last_conv_layer = model.get_layer(last_conv_layer_name)
    conv_model = tf.keras.models.Model(model.inputs, last_conv_layer.output)

    # Classifier model that takes conv output as input
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = tf.keras.models.Model(classifier_input, x)

    #Gradient calculation
    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_array)
        tape.watch(conv_outputs)
        preds = classifier_model(conv_outputs)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, conv_outputs)

    # Average the gradients over all spatial locations
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    conv_outputs = conv_outputs.numpy()[0]

    # Weight the conv layer activations by the averaged gradients
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]

    # Generate el heatmap
    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1
    heatmap = np.uint8(255 * heatmap)
    heatmap = np.expand_dims(heatmap, axis=-1)
    heatmap = tf.image.resize(heatmap, size).numpy()
    heatmap = heatmap[:, :, 0]

    #  Convert to 3-channel for compatibility with array_to_img
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=-1)
    return heatmap
