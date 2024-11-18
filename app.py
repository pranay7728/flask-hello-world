from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_path = "best_model.keras"  # Path to your saved model
best_model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']  # Update if needed

def classify_image(image_path):
    """
    Classify a single image.
    Args:
        image_path (str): Path to the image file.
    Returns:
        dict: Predicted class label and confidence.
    """
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 150))  # Resize to model input size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Rescale pixel values to [0, 1]

    # Predict
    predictions = best_model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class]

    return {"class": class_labels[predicted_class], "confidence": float(confidence)}

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for image classification.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg')):
        # Save the uploaded file
        temp_path = os.path.join("temp", file.filename)
        os.makedirs("temp", exist_ok=True)  # Ensure temp folder exists
        file.save(temp_path)

        # Perform prediction
        result = classify_image(temp_path)

        # Delete the temporary file
        os.remove(temp_path)

        return jsonify(result), 200
    else:
        return jsonify({"error": "Allowed file types are png, jpg, jpeg"}), 400

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
