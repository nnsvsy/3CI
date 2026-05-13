import sys
f = open('d:/第三次/minimal_log.txt', 'w')
f.write("Starting...\n")
f.flush()

try:
    import pandas as pd
    f.write(f"Pandas: {pd.__version__}\n")
    f.flush()
except Exception as e:
    f.write(f"Pandas error: {e}\n")
    f.flush()

try:
    import numpy as np
    f.write(f"NumPy: {np.__version__}\n")
    f.flush()
except Exception as e:
    f.write(f"NumPy error: {e}\n")
    f.flush()

try:
    f.write("Loading train.csv...\n")
    f.flush()
    train = pd.read_csv('d:/第三次/train.csv')
    f.write(f"Train shape: {train.shape}\n")
    f.flush()
except Exception as e:
    f.write(f"Train error: {e}\n")
    f.flush()

try:
    f.write("Loading test.csv...\n")
    f.flush()
    test = pd.read_csv('d:/第三次/test.csv')
    f.write(f"Test shape: {test.shape}\n")
    f.flush()
except Exception as e:
    f.write(f"Test error: {e}\n")
    f.flush()

try:
    from sklearn.ensemble import ExtraTreesClassifier
    f.write("Training ExtraTrees...\n")
    f.flush()

    X = train.drop('label', axis=1).values[:10000] / 255.0
    y = train['label'].values[:10000]
    X_test = test.values / 255.0

    clf = ExtraTreesClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
    clf.fit(X, y)
    f.write("Training done!\n")
    f.flush()

    pred = clf.predict(X_test)
    f.write(f"Predictions: {len(pred)}\n")
    f.flush()

    import pandas as pd
    sub = pd.DataFrame({'ImageId': range(1, len(pred)+1), 'Label': pred})
    sub.to_csv('d:/第三次/submission.csv', index=False)
    f.write("Saved!\n")
    f.flush()

except Exception as e:
    f.write(f"Error: {e}\n")
    import traceback
    f.write(traceback.format_exc())
    f.flush()

f.close()
print("Done. Check minimal_log.txt")