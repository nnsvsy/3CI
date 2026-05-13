import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(filename='d:/第三次/lgb_log.txt', level=logging.INFO, format='%(message)s')
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

    logger.info("Training LightGBM...")
    params = {
        'objective': 'multiclass',
        'num_class': 10,
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }

    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30),
            lgb.log_evaluation(period=50)
        ]
    )

    logger.info(f"Best iteration: {model.best_iteration}")

    val_pred = model.predict(X_val)
    val_pred_labels = np.argmax(val_pred, axis=1)
    val_acc = np.mean(val_pred_labels == y_val)
    logger.info(f"Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")

    logger.info("Predicting...")
    test_pred = model.predict(X_test)
    final_pred = np.argmax(test_pred, axis=1)

    sub = pd.DataFrame({'ImageId': range(1, len(final_pred)+1), 'Label': final_pred})
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