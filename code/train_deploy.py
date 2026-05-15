import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

print("="*60)
print("MNIST CNN Model Training")
print("姓名：佴悦")
print("学号：112311170312")
print("="*60)

print("\n[1/6] Loading data...")
train_df = pd.read_csv('d:/第三次/train.csv')
test_df = pd.read_csv('d:/第三次/test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_df['label'].values
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

print("\n[2/6] Building CNN model...")
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("\n[3/6] Training model...")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

print("\n[4/6] Saving model...")
model.save('d:/第三次/code/best_cnn.h5')
print("Model saved to: d:/第三次/code/best_cnn.h5")

print("\n[5/6] Making predictions...")
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)

print("\n[6/6] Creating submission...")
submission = pd.DataFrame({
    'ImageId': range(1, len(pred_labels) + 1),
    'Label': pred_labels
})
submission.to_csv('d:/第三次/results/submission.csv', index=False)
print(f"Submission saved to: d:/第三次/results/submission.csv")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
print("="*60)