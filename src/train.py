import os
import pandas as pd
from src.data_processing import load_data, create_time_features, add_lag_and_rolling, build_label_by_threshold
from src.models import train_xgboost, train_lstm
import numpy as np


DATA_PATH = os.path.join('data', 'flood_data.csv')
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)




def prepare_tabular(df):
    # choose some sensible default features (modify according to your CSV)
    # priority: keep numeric columns only
    drop_cols = ['date'] if 'date' in df.columns else []
    features = [c for c in df.columns if c not in drop_cols + ['flood_label', 'flood_threshold_used']]
    X = df[features].select_dtypes(include=[np.number]).fillna(0)
    y = df['flood_label']
    return X, y




def prepare_sequences(df, features, seq_len=14):
    Xs = []
    ys = []
    for i in range(seq_len, len(df)):
        Xs.append(df[features].iloc[i-seq_len:i].values)
        ys.append(df['flood_label'].iloc[i])
    Xs = np.array(Xs)
    ys = np.array(ys)
    return Xs, ys




if __name__ == '__main__':
    print('Loading data...')
    df = load_data(DATA_PATH)
    print('Creating time features...')
    df = create_time_features(df)
    df = add_lag_and_rolling(df, col='rainfall_mm', lags=[1,3,7], windows=[3,7,14])


    # If flood_label not present, build it
    if 'flood_label' not in df.columns:
        df = build_label_by_threshold(df, level_col='river_level', threshold=None)


    # Train XGBoost (tabular)
    X, y = prepare_tabular(df)
    print('Training XGBoost...')
    xgb_model = train_xgboost(X, y, save_path=os.path.join(MODEL_DIR, 'xgb_model.joblib'))


    # Train LSTM (sequence)
    numeric_features = X.columns.tolist()
    print('Preparing sequences for LSTM...')
    X_seq, y_seq = prepare_sequences(df, numeric_features, seq_len=14)
    print('Training LSTM...')
    lstm_model = train_lstm(X_seq, y_seq, save_path=os.path.join(MODEL_DIR, 'lstm_model'))


    print('Training complete. Models saved to', MODEL_DIR)