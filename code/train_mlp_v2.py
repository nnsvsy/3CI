import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename='d:/第三次/mlp_log.txt', level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

logger.info("=" * 50)
logger.info("Script starting...")

try:
    logger.info("Loading data...")
    train = pd.read_csv('d:/第三次/train.csv')
    test = pd.read_csv('d:/第三次/test.csv')
    logger.info(f"Train: {train.shape}, Test: {test.shape}")

    X = train.drop('label', axis=1).values / 255.0
    y = train['label'].values
    X_test = test.values / 255.0

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    logger.info(f"Training: {X_tr.shape}, Validation: {X_val.shape}")

    logger.info("Training MLP (512-256-128)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256, 128),
        activation='relu',
        solver='adam',
        alpha=0.00001,
        batch_size=256,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=50,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=True
    )
    mlp.fit(X_tr, y_tr)

    val_pred = mlp.predict(X_val)
    val_acc = np.mean(val_pred == y_val)
    logger.info(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    logger.info("Predicting...")
    pred = mlp.predict(X_test)

    sub = pd.DataFrame({'ImageId': range(1, len(pred)+1), 'Label': pred})
    sub.to_csv('d:/第三次/submission.csv', index=False)
    logger.info("Submission saved!")
    logger.info(f"Predictions:\n{sub.head(10)}")
    logger.info("DONE!")

    print(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
    print("Submission saved!")
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    logger.error(traceback.format_exc())
    print(f"Error: {e}")