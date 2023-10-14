import tensorflow as tf
import tf2onnx
import onnx
from keras.models import load_model

# Load your Keras model
keras_model = load_model('Models/action_test_2.h5')
keras_model.summary()

# Convert the Keras model to ONNX format without specifying input_signature and opset
onnx_model, _ = tf2onnx.convert.from_keras(keras_model)

# Save the ONNX model to a file
onnx.save(onnx_model, "Models/action_test_2.onnx")