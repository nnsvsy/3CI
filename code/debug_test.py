# Simple test script
import sys
print("="*60)
print("PYTHON TEST")
print("="*60)
print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print(f"CWD: {__file__}")

# Quick imports
try:
    import pandas as pd
    print(f"Pandas: {pd.__version__} - OK")
except Exception as e:
    print(f"Pandas: FAIL - {e}")

try:
    import numpy as np
    print(f"NumPy: {np.__version__} - OK")
except Exception as e:
    print(f"NumPy: FAIL - {e}")

try:
    import torch
    print(f"PyTorch: {torch.__version__} - OK")
except Exception as e:
    print(f"PyTorch: FAIL - {e}")

try:
    import sklearn
    print(f"sklearn: {sklearn.__version__} - OK")
except Exception as e:
    print(f"sklearn: FAIL - {e}")

print("\nTesting data load...")
try:
    train = pd.read_csv('d:/第三次/train.csv')
    test = pd.read_csv('d:/第三次/test.csv')
    print(f"Train: {train.shape}, Test: {test.shape} - OK")
except Exception as e:
    print(f"Data load: FAIL - {e}")

print("="*60)
print("TEST COMPLETE")
print("="*60)