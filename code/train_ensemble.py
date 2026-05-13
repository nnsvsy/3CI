import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

print("Loading data...")
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

X_train = train.drop('label', axis=1).values / 255.0
y_train = train['label'].values
X_test = test.values / 255.0

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
print(f"Training set: {X_tr.shape}, Validation set: {X_val.shape}")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=25, n_jobs=-1, random_state=42, verbose=1)
rf.fit(X_tr, y_tr)
rf_val_acc = rf.score(X_val, y_val)
print(f"Random Forest Validation Accuracy: {rf_val_acc:.4f}")

print("\nTraining Extra Trees...")
et = ExtraTreesClassifier(n_estimators=300, max_depth=25, n_jobs=-1, random_state=42, verbose=1)
et.fit(X_tr, y_tr)
et_val_acc = et.score(X_val, y_val)
print(f"Extra Trees Validation Accuracy: {et_val_acc:.4f}")

print("\nTraining MLP...")
mlp = MLPClassifier(
    hidden_layer_sizes=(512, 256, 128),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    batch_size=128,
    learning_rate='adaptive',
    max_iter=50,
    random_state=42,
    verbose=True
)
mlp.fit(X_tr, y_tr)
mlp_val_acc = mlp.score(X_val, y_val)
print(f"MLP Validation Accuracy: {mlp_val_acc:.4f}")

print("\nCreating ensemble prediction...")
rf_pred = rf.predict(X_test)
et_pred = et.predict(X_test)
mlp_pred = mlp.predict(X_test)

from scipy import stats
ensemble_pred = np.array([rf_pred, et_pred, mlp_pred])
final_pred = stats.mode(ensemble_pred, axis=0)[0].flatten()

submission = pd.DataFrame({
    'ImageId': range(1, len(final_pred) + 1),
    'Label': final_pred.astype(int)
})

submission.to_csv('d:/第三次/submission.csv', index=False)
print("Submission saved!")
print(submission.head(10))