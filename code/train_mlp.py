import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

X_train = train.drop('label', axis=1).values / 255.0
y_train = train['label'].values
X_test = test.values / 255.0

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

print("Training MLP classifier...")
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=128,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=100,
    shuffle=True,
    random_state=42,
    verbose=True,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=10
)

mlp.fit(X_train, y_train)

val_pred = mlp.predict(X_val)
val_acc = np.mean(val_pred == y_val)
print(f"\nValidation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

print("\nMaking predictions...")
predictions = mlp.predict(X_test)

submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})

submission.to_csv('d:/第三次/submission.csv', index=False)
print("Submission saved to d:/第三次/submission.csv")
print(submission.head(10))