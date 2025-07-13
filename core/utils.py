import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import yaml

from core.embeddings import get_tfidf_embd

with open("configs/config.yaml", "r") as f:
    Config = yaml.safe_load(f)
import random
seed =0
random.seed(seed)
np.random.seed(seed)

class Data():
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        # Store input features and labels
        self.X = X
        self.y = y
        self.y_series = pd.Series(y)  # Convert labels to pandas Series for easier analysis

        # Identify class labels that appear at least 3 times
        good_y_value = self.y_series.value_counts()[self.y_series.value_counts() >= 3].index

        # If no class has 3 or more samples, skip processing
        if len(good_y_value) < 1:
            print("None of the classes have more than 3 records: Skipping ...")
            self.X_train = None
            self.X_test = None
            self.y_train = None
            self.y_test = None
            self.classes = None
            self.embeddings = None
            return

        # Filter the dataset to only include samples from valid classes
        y_good = y[self.y_series.isin(good_y_value)]
        X_good = X[self.y_series.isin(good_y_value)]

        # Recalculate test set size so it's approximately 20% of the original data
        new_test_size = X.shape[0] * 0.2 / X_good.shape[0]

        # Split filtered data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, test_size=new_test_size, random_state=seed, stratify=y_good
        )

        # Store the valid class labels and original embeddings
        self.classes = good_y_value
        self.embeddings = X

    # Get original labels
    def get_type(self):
        return self.y

    # Get training features
    def get_X_train(self):
        return self.X_train

    # Get testing features
    def get_X_test(self):
        return self.X_test

    # Get training labels
    def get_type_y_train(self):
        return self.y_train

    # Get testing labels
    def get_type_y_test(self):
        return self.y_test

    # Get training dataframe (not initialized here; must be set externally)
    def get_train_df(self):
        return self.train_df

    # Get original embeddings
    def get_embeddings(self):
        return self.embeddings

    # Get testing dataframe (not initialized here; must be set externally)
    def get_type_test_df(self):
        return self.test_df

    # Get test embeddings for deep learning (not initialized here)
    def get_X_DL_test(self):
        return self.X_DL_test

    # Get train embeddings for deep learning (not initialized here)
    def get_X_DL_train(self):
        return self.X_DL_train


def prepare_data(df):
    # Split the data for each label target
    
    X = get_tfidf_embd(df)
    y_intent = df['y2'].to_numpy()
    y_tone = df['y3'].to_numpy()
    y_resolution = df['y4'].to_numpy()

    X_train, X_test, y_intent_train, y_intent_test = train_test_split(X, y_intent, test_size=0.2, random_state=42)
    _, _, y_tone_train, y_tone_test = train_test_split(X, y_tone, test_size=0.2, random_state=42)
    _, _, y_resolution_train, y_resolution_test = train_test_split(X, y_resolution, test_size=0.2, random_state=42)

    return X_train, X_test, y_intent_train, y_intent_test, y_tone_train, y_tone_test, y_resolution_train, y_resolution_test

# visualization functions
def plot_confusion(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def plot_class_distribution(df, label_col, title="Class Distribution"):
    plt.figure(figsize=(10, 4))
    sns.countplot(x=label_col, data=df)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_classification_report(y_true, y_pred, title="Classification Report"):
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:, :-1], annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title(title)
    plt.show()