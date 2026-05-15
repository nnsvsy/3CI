import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("MNIST CNN Training - 佴悦 112311170312")
print("="*60)

print("\n[1/5] Loading data...")
train_df = pd.read_csv('d:/第三次/train.csv')
test_df = pd.read_csv('d:/第三次/test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_df['label'].values
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

print("\n[2/5] Building model...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n[3/5] Training (15 epochs)...")
model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1, verbose=1)

print("\n[4/5] Saving model...")
model.save('d:/第三次/code/best_cnn.h5')
print("Model saved!")

print("\n[5/5] Creating submission...")
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)

submission = pd.DataFrame({'ImageId': range(1, len(pred_labels) + 1), 'Label': pred_labels})
submission.to_csv('d:/第三次/results/submission.csv', index=False)

print("\n" + "="*60)
print(f"Training Accuracy: {model.evaluate(X_train, y_train, verbose=0)[1]:.4f}")
print("="*60)
print("DONE! Model and submission saved.")