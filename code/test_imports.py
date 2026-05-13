import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"PyTorch import error: {e}")

try:
    import pandas as pd
    print(f"Pandas version: {pd.__version__}")
except Exception as e:
    print(f"Pandas import error: {e}")

try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except Exception as e:
    print(f"NumPy import error: {e}")

try:
    from sklearn.model_selection import train_test_split
    print("sklearn imported successfully")
except Exception as e:
    print(f"sklearn import error: {e}")

print("\nAll imports successful!")