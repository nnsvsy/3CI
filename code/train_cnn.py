import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
import os

log_file = open('d:/第三次/training_log.txt', 'w')

def log_print(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, **kwargs, file=log_file, flush=True)

log_print(f"PyTorch version: {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log_print(f"Using device: {device}")

log_print("Loading data...")
train = pd.read_csv('d:/第三次/train.csv')
test = pd.read_csv('d:/第三次/test.csv')

log_print(f"Train shape: {train.shape}")
log_print(f"Test shape: {test.shape}")

X_train = train.drop('label', axis=1).values.reshape(-1, 1, 28, 28).astype('float32') / 255.0
y_train = train['label'].values
X_test = test.values.reshape(-1, 1, 28, 28).astype('float32') / 255.0

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
log_print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_val_t = torch.FloatTensor(X_val)
y_val_t = torch.LongTensor(y_val)
X_test_t = torch.FloatTensor(X_test)

train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset = TensorDataset(X_val_t, y_val_t)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        if batch_idx % 100 == 0:
            log_print(f"  Batch {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
    return total_loss / len(loader), correct / total

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

log_print("\nTraining CNN model...")
best_val_acc = 0
patience_counter = 0
max_patience = 10

for epoch in range(30):
    log_print(f"\nEpoch {epoch+1}/30")
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_acc = evaluate(model, val_loader, device)
    scheduler.step(val_acc)

    log_print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'd:/第三次/best_model.pth')
        log_print(f"  New best model saved! Val Acc: {val_acc:.4f}")
        patience_counter = 0
    else:
        patience_counter += 1
        log_print(f"  No improvement. Patience: {patience_counter}/{max_patience}")
        if patience_counter >= max_patience:
            log_print(f"Early stopping at epoch {epoch+1}")
            break

log_print(f"\nBest Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")

if best_val_acc > 0:
    model.load_state_dict(torch.load('d:/第三次/best_model.pth'))

log_print("\nMaking predictions...")
model.eval()
with torch.no_grad():
    X_test_t = X_test_t.to(device)
    outputs = model(X_test_t)
    _, predicted_labels = torch.max(outputs.data, 1)
    predicted_labels = predicted_labels.cpu().numpy()

submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})

submission.to_csv('d:/第三次/submission.csv', index=False)
log_print("Submission saved to d:/第三次/submission.csv")
log_print(submission.head(10))

log_file.close()
print("Training complete! Check training_log.txt for details.")