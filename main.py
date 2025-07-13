from core.embeddings import get_tfidf_embd
from core.preprocess import get_input_data, de_duplication, noise_remover, translate_to_en, Config, label_encoder
from core.utils import Data, prepare_data

from pipeline.trainer import model_predict

import pandas as pd
import numpy as np
import random
seed =0
random.seed(seed)
np.random.seed(seed)

def load_data():
    # Load your data here
    return get_input_data()

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
    # De-duplicate input data
    df =  de_duplication(df)
    # remove noise in input data
    df = noise_remover(df)

    # Handle missing values in the target columns
    df = df.dropna(subset=[Config['INTENT_COL'], Config['TONE_COL'], Config['RESOLUTION_COL']])

    # Encode categorical target variables into numeric values
    label_encoders = label_encoder(df)

    # Remove rows with empty text data
    df = df[(df[Config['TICKET_SUMMARY']].str.strip() != '') & (df[Config['INTERACTION_CONTENT']].str.strip() != '')]

    # translate data to english
    # df[Config['TICKET_SUMMARY']] = translate_to_en(df[Config['TICKET_SUMMARY']].tolist())
    return df, label_encoders

def get_embeddings(df:pd.DataFrame):
    X = get_tfidf_embd(df)  # get tf-idf embeddings
    return X, df

def get_data_object(X: np.ndarray, df: pd.DataFrame):
    return Data(X, df)

def perform_modelling(data: Data, df: pd.DataFrame, name):
    model_predict(data, df, name)

if __name__ == '__main__':
    df = load_data()
    df, label_encoders = preprocess_data(df)
    df[Config['INTERACTION_CONTENT']] = df[Config['INTERACTION_CONTENT']].values.astype('U')
    df[Config['TICKET_SUMMARY']] = df[Config['TICKET_SUMMARY']].values.astype('U')
    
    X_train, X_test, y_intent_train, y_intent_test, y_tone_train, y_tone_test, y_resolution_train, y_resolution_test = prepare_data(
        df)
    
    # for name, group_df in grouped_df:
    #     print(name)
    #     X, group_df = get_embeddings(group_df)
    #     data = get_data_object(X, group_df)
    #     perform_modelling(data, group_df, name)