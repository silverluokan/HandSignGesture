# tflite_converter.py

import tensorflow as tf

def convert_keras_to_tflite(keras_model_path, tflite_model_path):
    """
    Converts a Keras model to TensorFlow Lite format.

    Args:
    keras_model_path (str): Path to the Keras model file.
    tflite_model_path (str): Path where the converted TFLite model will be saved.
    """
    # Load the Keras model
    model = tf.keras.models.load_model(keras_model_path)

    # Convert the model to TFLite format with select TensorFlow ops enabled
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model converted and saved to {tflite_model_path}")
