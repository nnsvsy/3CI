import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import sys

print("Starting script...", flush=True)

try:
    print("Loading data...", flush=True)
    train = pd.read_csv('d:/第三次/train.csv')
    test = pd.read_csv('d:/第三次/test.csv')
    print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

    X = train.drop('label', axis=1).values / 255.0
    y = train['label'].values
    X_test = test.values / 255.0

    print("Training Random Forest (300 trees)...", flush=True)
    clf = RandomForestClassifier(n_estimators=300, max_depth=30, n_jobs=-1, random_state=42, verbose=1)
    clf.fit(X, y)

    print("Predicting...", flush=True)
    pred = clf.predict(X_test)

    sub = pd.DataFrame({'ImageId': range(1, len(pred)+1), 'Label': pred})
    sub.to_csv('d:/第三次/submission.csv', index=False)
    print("Done! Submission saved.", flush=True)
    print(sub.head(), flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()