from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

print("Loading model...")
model = tf.keras.models.load_model('best_cnn.h5')
print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        img = Image.open(io.BytesIO(image_bytes)).convert('L')
        img = img.resize((28, 28))
        img_array = np.array(img).reshape(1, 28, 28, 1).astype('float32') / 255.0
        
        prediction = model.predict(img_array)
        digit = np.argmax(prediction)
        confidence = float(prediction[0][digit]) * 100
        
        return jsonify({'digit': int(digit), 'confidence': round(confidence, 2)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)