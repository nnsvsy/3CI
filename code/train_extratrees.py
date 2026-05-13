# Minimal MNIST training script
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import pickle

print("Loading data...")
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}")

X = train.drop('label', axis=1).values / 255.0
y = train['label'].values
X_test = test.values / 255.0

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Train: {X_tr.shape}, Val: {X_val.shape}")

print("Training ExtraTrees (500 trees)...")
clf = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=42, verbose=1)
clf.fit(X_tr, y_tr)

val_acc = clf.score(X_val, y_val)
print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

print("Training on full data...")
clf_full = ExtraTreesClassifier(n_estimators=500, max_depth=None, min_samples_split=2, n_jobs=-1, random_state=42)
clf_full.fit(X, y)

print("Predicting...")
pred = clf_full.predict(X_test)

sub = pd.DataFrame({'ImageId': range(1, len(pred)+1), 'Label': pred})
sub.to_csv('d:/第三次/submission.csv', index=False)
print("Saved submission.csv")
print(sub.head(10))