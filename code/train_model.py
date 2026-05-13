import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

print("Loading data...")
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

X_train = train.drop('label', axis=1).values
y_train = train['label'].values
X_test = test.values

X_train = X_train / 255.0
X_test = X_test / 255.0

print("Training model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    n_jobs=-1,
    random_state=42,
    verbose=1
)
model.fit(X_train, y_train)

print("Making predictions...")
predictions = model.predict(X_test)

submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})

submission.to_csv('d:/第三次/submission.csv', index=False)
print("Submission saved to d:/第三次/submission.csv")
print(submission.head(10))