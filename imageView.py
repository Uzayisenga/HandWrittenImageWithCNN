from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode y_test
y_test_encoded = tf.keras.utils.to_categorical(y_test, num_classes=10)

# Ensure the model path is correct
model_path = 'mnist_cnn_model.h5'
try:
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")

# Evaluate the model on the entire test set
test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

@app.route('/')
def index():
    return render_template('index.html', accuracy=test_accuracy, loss=test_loss)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(image_file)
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_digit = np.argmax(prediction)

        # Evaluate the model on the processed image itself
        loss, accuracy = model.evaluate(processed_image, tf.keras.utils.to_categorical([predicted_digit], num_classes=10), verbose=0)

        return jsonify({'digit': int(predicted_digit), 'accuracy': accuracy, 'loss': loss})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).astype('float32') / 255.0
    image = image.reshape((1, 28, 28, 1))
    return image

if __name__ == '__main__':
    app.run(debug=True)

