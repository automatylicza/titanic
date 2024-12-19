import pandas as pd
from pathlib import Path

def load_data():
    base_dir = Path.cwd().parent
    train_path = base_dir / "data" / "external" / "train.csv"
    test_path = base_dir / "data" / "external" / "test.csv"
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test