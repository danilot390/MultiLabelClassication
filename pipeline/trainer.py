from models.multi_label_rf import RandomForest
from core.utils import Data
import numpy as np 

def model_predict(data, df, name):
    results = []
    print("RandomForest")
    model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
    model.train(data)
    model.predict(data.X_test)
    model.print_results(data)


def model_evaluate(model, data):
    model.print_results(data)

def chained_model_training(X_train, y_intent_train, y_tone_train, y_resolution_train):
    # Model A: Predict Intent
    data_a = Data(X_train, y_intent_train)
    model_a = RandomForest("Model_A", X_train, y_intent_train)
    model_a.train(data_a)

    # Ensure intent_predictions is not None
    intent_predictions = model_a.predict(X_train)
    if intent_predictions is None:
        raise ValueError("Intent predictions are None")

    X_train_with_intent = np.hstack((X_train, intent_predictions.reshape(-1, 1)))

    # Model B: Predict Tone using Intent predictions
    data_b = Data(X_train_with_intent, y_tone_train)
    model_b = RandomForest("Model_B", X_train_with_intent, y_tone_train)
    model_b.train(data_b)

    # Ensure tone_predictions is not None
    tone_predictions = model_b.predict(X_train_with_intent)
    if tone_predictions is None:
        raise ValueError("Tone predictions are None")

    X_train_with_tone = np.hstack((X_train_with_intent, tone_predictions.reshape(-1, 1)))

    # Model C: Predict Resolution using Intent and Tone predictions
    data_c = Data(X_train_with_tone, y_resolution_train)
    model_c = RandomForest("Model_C", X_train_with_tone, y_resolution_train)
    model_c.train(data_c)

    return model_a, model_b, model_c

def chained_model_prediction(model_a, model_b, model_c, X_test):
    # Predict Intent
    intent_predictions = model_a.predict(X_test)

    # Predict Tone using Intent predictions
    X_test_with_intent = np.hstack((X_test, intent_predictions.reshape(-1, 1)))
    tone_predictions = model_b.predict(X_test_with_intent)

    # Predict Resolution using Intent and Tone predictions
    X_test_with_tone = np.hstack((X_test_with_intent, tone_predictions.reshape(-1, 1)))
    resolution_predictions = model_c.predict(X_test_with_tone)

    return intent_predictions, tone_predictions, resolution_predictions