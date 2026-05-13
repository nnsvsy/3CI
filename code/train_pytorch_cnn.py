import sys
import os

print("=" * 60, flush=True)
print("MNIST CNN Training with PyTorch", flush=True)
print("=" * 60, flush=True)

sys.stdout.flush()

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print(f"PyTorch: {torch.__version__}", flush=True)
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}", flush=True)

print("\n[1/6] Loading data...", flush=True)
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')
print(f"Train: {train.shape}, Test: {test.shape}", flush=True)

print("\n[2/6] Preprocessing...", flush=True)
X_train = train.drop('label', axis=1).values.reshape(-1, 1, 28, 28).astype('float32') / 255.0
y_train = train['label'].values
X_test = test.values.reshape(-1, 1, 28, 28).astype('float32') / 255.0
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}", flush=True)

val_size = int(0.1 * len(X_train))
X_val = X_train[:val_size]
y_val = y_train[:val_size]
X_train = X_train[val_size:]
y_train = y_train[val_size:]
print(f"Train: {X_train.shape}, Val: {X_val.shape}", flush=True)

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)

train_ds = TensorDataset(X_train_t, y_train_t)
val_ds = TensorDataset(X_val_t, y_val_t)

batch_size = 256
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

print("\n[3/6] Building CNN model...", flush=True)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN().to(device)
print(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

print("\n[4/6] Training...", flush=True)

best_val_acc = 0.0
for epoch in range(15):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_acc = 100. * correct / total

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_acc = 100. * correct / total
    scheduler.step()

    print(f"Epoch {epoch+1:2d}/15 - Train: {train_acc:.2f}% - Val: {val_acc:.2f}%", flush=True)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'd:/第三次/best_cnn.pth')

print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%", flush=True)

print("\n[5/6] Loading best model...", flush=True)
model.load_state_dict(torch.load('d:/第三次/best_cnn.pth'))

print("\n[6/6] Generating predictions...", flush=True)
model.eval()
with torch.no_grad():
    X_test_t = X_test_t.to(device)
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    predictions = predicted.cpu().numpy()

submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})
submission.to_csv('d:/第三次/submission.csv', index=False)

print("\n" + "=" * 60, flush=True)
print(f"DONE! Best Val Acc: {best_val_acc:.2f}%", flush=True)
print("Submission saved to d:/第三次/submission.csv", flush=True)
print("=" * 60, flush=True)