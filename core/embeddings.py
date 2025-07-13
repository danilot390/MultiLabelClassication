import numpy as np
import pandas as pd
import yaml

with open("configs/config.yaml", "r") as f:
    Config = yaml.safe_load(f)

import random
seed =0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df:pd.DataFrame):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)
    data = df[Config['TICKET_SUMMARY']] + ' ' + df[Config['INTERACTION_CONTENT']]
    X = tfidfconverter.fit_transform(data).toarray()
    return X
