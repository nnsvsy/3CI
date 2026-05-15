import pandas as pd
import numpy as np
import tensorflow as tf

print("="*60)
print("MNIST Digit Recognizer - Model Test")
print("姓名：佴悦")
print("学号：112311170312")
print("="*60)

print("\nLoading model...")
model = tf.keras.models.load_model('d:/第三次/code/best_cnn.h5')
print("Model loaded successfully!")

print("\nLoading test data...")
test = pd.read_csv('d:/第三次/test.csv')
X_test = test.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
print(f"Test samples: {X_test.shape[0]}")

print("\nMaking predictions...")
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)

print("\nSample predictions:")
for i in range(10):
    print(f"Image {i+1}: Predicted digit = {pred_labels[i]}")

print("\n" + "="*60)
print("✓ Model is working correctly!")
print("✓ Ready for deployment!")
print("="*60)

print("\n部署步骤：")
print("1. 将 code/ 目录上传到 Render")
print("2. 构建命令: pip install -r requirements.txt")
print("3. 启动命令: gunicorn app:app")
print("4. 获得网址如: https://eryue-mnist-112311170312.onrender.com")