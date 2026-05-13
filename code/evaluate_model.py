import pandas as pd
import numpy as np
import tensorflow as tf

print("Loading model...")
model = tf.keras.models.load_model('d:/第三次/best_cnn.h5')

print("Loading data...")
train = pd.read_csv('d:/第三次/train.csv')
X = train.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y = train['label'].values
y_onehot = tf.keras.utils.to_categorical(y, 10)

print("Evaluating model on training data...")
loss, acc = model.evaluate(X, y_onehot, verbose=0)
print(f"Training Accuracy: {acc:.4f} ({acc*100:.2f}%)")

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y_onehot, test_size=0.1, random_state=42)

print("Evaluating model on validation data...")
val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

print("\n" + "="*60)
if val_acc >= 0.99:
    print(f"✓ SUCCESS: Accuracy exceeds 99% ({val_acc*100:.2f}%)")
else:
    print(f"✗ Need improvement: Accuracy is {val_acc*100:.2f}%")
print("="*60)