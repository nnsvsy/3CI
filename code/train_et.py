import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
import sys

print("Step 1: Loading data...", flush=True)
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')
print(f"  Train: {train.shape}, Test: {test.shape}", flush=True)

X = train.drop('label', axis=1).values / 255.0
y = train['label'].values
X_test = test.values / 255.0

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"Step 2: Split data - Train: {X_tr.shape}, Val: {X_val.shape}", flush=True)

print("Step 3: Training ExtraTrees classifier...", flush=True)
clf = ExtraTreesClassifier(n_estimators=300, max_depth=30, n_jobs=-1, random_state=42, verbose=0)
clf.fit(X_tr, y_tr)
print("  Training complete!", flush=True)

val_acc = clf.score(X_val, y_val)
print(f"Step 4: Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)", flush=True)

if val_acc >= 0.97:
    print("Step 5: Retraining on full data...", flush=True)
    clf_full = ExtraTreesClassifier(n_estimators=300, max_depth=30, n_jobs=-1, random_state=42)
    clf_full.fit(X, y)

    print("Step 6: Generating predictions...", flush=True)
    pred = clf_full.predict(X_test)

    submission = pd.DataFrame({'ImageId': range(1, len(pred)+1), 'Label': pred})
    submission.to_csv('d:/第三次/submission.csv', index=False)
    print("Step 7: Submission saved!", flush=True)
    print(submission.head(10), flush=True)
else:
    print("Accuracy too low, need better model", flush=True)