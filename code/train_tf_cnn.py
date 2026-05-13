import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

print("\n[1/5] Loading data...")
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

X = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y = train['label'].values
X_test = test.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y = tf.keras.utils.to_categorical(y, 10)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}")

print("\n[2/5] Building CNN model...")
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\n[3/5] Training model...")
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('d:/第三次/best_cnn.h5', monitor='val_accuracy', save_best_only=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6)

history = model.fit(
    X_train, y_train,
    batch_size=128,
    epochs=50,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint, lr_scheduler],
    verbose=1
)

print("\n[4/5] Evaluating...")
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

print("\n[5/5] Generating predictions...")
predictions = model.predict(X_test)
pred_labels = np.argmax(predictions, axis=1)

submission = pd.DataFrame({'ImageId': range(1, len(pred_labels)+1), 'Label': pred_labels})
submission.to_csv('d:/第三次/submission.csv', index=False)

print("\n" + "="*60)
print(f"Final Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
print("Submission saved to d:/第三次/submission.csv")
print("="*60)