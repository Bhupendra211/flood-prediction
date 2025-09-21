import joblib
import numpy as np
import pandas as pd
import tensorflow as tf




def load_xgb_model(path: str):
    return joblib.load(path)




def load_lstm_model(path: str):
    return tf.keras.models.load_model(path)




def predict_tabular(model, X_row: pd.DataFrame):
    proba = model.predict_proba(X_row)[:,1]
    return proba




def predict_lstm(model, seq: np.ndarray):
    # seq shape: (1, seq_len, features)
    proba = model.predict(seq)
    return proba