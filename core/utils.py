import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from pathlib import Path
import os

CONFIG_PATH = Path(os.path.join('configs', 'config.yaml'))
if not CONFIG_PATH.exists():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

def load_config(path: Path = CONFIG_PATH):
    """Load configuration from a YAML file."""
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)  
    
Config = load_config()

import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 df: pd.DataFrame,
                 min_samples_per_label: int =3,
                 verbose: bool = True) -> None:
        
        self.embeddings = X
        self.df = df
        self.y = y

        label_counts = y.sum(axis=0)
        valid_labels = label_counts >= min_samples_per_label

        if not valid_labels.any():
            if verbose:
                print('Sonthing goes wrong with labels')
            self.X_train = self.X_test = self.y_train, self.y_test = None

        


        y = df.y.to_numpy()
        y_series = pd.Series(y)

        good_y_value = y_series.value_counts()[y_series.value_counts() >= 3].index

        if len(good_y_value)<1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]

        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]


        self.X_train, self.X_test, self.y_train, self.y_test= train_test_split(X_good, y_good, test_size=new_test_size, random_state=0, stratify=y_good)
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X


    def get_type(self):
        return  self.y
    def get_X_train(self):
        return  self.X_train
    def get_X_test(self):
        return  self.X_test
    def get_type_y_train(self):
        return  self.y_train
    def get_type_y_test(self):
        return  self.y_test
    def get_train_df(self):
        return  self.train_df
    def get_embeddings(self):
        return  self.embeddings
    def get_type_test_df(self):
        return  self.test_df
    def get_X_DL_test(self):
        return self.X_DL_test
    def get_X_DL_train(self):
        return self.X_DL_train

def save_data(df: pd.DataFrame, path: Path = Path(os.path.join('data','processed','data.csv') )):
    """Save DataFrame to a CSV file."""
    # df.to_csv(path, index=False)
    print(df)
    print(f"Data saved to {path}")