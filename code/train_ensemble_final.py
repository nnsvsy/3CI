import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import sys

print("=" * 60)
print("MNIST Ensemble Classifier")
print("=" * 60)

print("\nStep 1: Loading data...")
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')
print(f"  Train: {train.shape}, Test: {test.shape}")

X = train.drop('label', axis=1).values / 255.0
y = train['label'].values
X_test = test.values / 255.0

X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
print(f"  Train: {X_tr.shape}, Val: {X_val.shape}")

print("\nStep 2: Training Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
rf_acc = rf.score(X_val, y_val)
print(f"  Validation Accuracy: {rf_acc:.4f}")

print("\nStep 3: Training ExtraTrees...")
et = ExtraTreesClassifier(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42)
et.fit(X_tr, y_tr)
et_acc = et.score(X_val, y_val)
print(f"  Validation Accuracy: {et_acc:.4f}")

print("\nStep 4: Training MLP...")
mlp = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
                    alpha=0.0001, batch_size=256, max_iter=30, random_state=42, verbose=0)
mlp.fit(X_tr, y_tr)
mlp_acc = mlp.score(X_val, y_val)
print(f"  Validation Accuracy: {mlp_acc:.4f}")

print("\nStep 5: Creating Voting Classifier...")
voting_clf = VotingClassifier(
    estimators=[('rf', rf), ('et', et), ('mlp', mlp)],
    voting='hard'
)
voting_clf.fit(X_tr, y_tr)
voting_acc = voting_clf.score(X_val, y_val)
print(f"  Validation Accuracy: {voting_acc:.4f} ({voting_acc*100:.2f}%)")

print("\nStep 6: Training on full dataset...")
print("  Training RF...")
rf_full = RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42)
rf_full.fit(X, y)

print("  Training ET...")
et_full = ExtraTreesClassifier(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42)
et_full.fit(X, y)

print("  Training MLP...")
mlp_full = MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu', solver='adam',
                         alpha=0.0001, batch_size=256, max_iter=30, random_state=42)
mlp_full.fit(X, y)

print("\nStep 7: Ensemble predictions...")
rf_pred = rf_full.predict(X_test)
et_pred = et_full.predict(X_test)
mlp_pred = mlp_full.predict(X_test)

from scipy import stats
final_pred, _ = stats.mode([rf_pred, et_pred, mlp_pred], axis=0)
final_pred = final_pred.flatten()

print("\nStep 8: Saving submission...")
submission = pd.DataFrame({'ImageId': range(1, len(final_pred)+1), 'Label': final_pred})
submission.to_csv('d:/第三次/submission.csv', index=False)
print("  Done!")

print("\n" + "=" * 60)
print(f"Ensemble Validation Accuracy: {voting_acc:.4f} ({voting_acc*100:.2f}%)")
print("Submission saved to submission.csv")
print("=" * 60)