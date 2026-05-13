import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(filename='d:/第三次/run_log.txt', level=logging.INFO, format='%(message)s')
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

    logger.info("Training Random Forest (200 trees, depth 30)...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=30, n_jobs=-1, random_state=42, verbose=0)
    clf.fit(X, y)
    logger.info("Training complete!")

    train_acc = clf.score(X, y)
    logger.info(f"Training accuracy: {train_acc:.4f}")

    logger.info("Predicting...")
    pred = clf.predict(X_test)

    sub = pd.DataFrame({'ImageId': range(1, len(pred)+1), 'Label': pred})
    sub.to_csv('d:/第三次/submission.csv', index=False)
    logger.info("Submission saved!")
    logger.info(f"Predictions:\n{sub.head(10)}")
    logger.info("DONE!")
except Exception as e:
    logger.error(f"Error: {e}")
    import traceback
    logger.error(traceback.format_exc())

print("Check run_log.txt for output")