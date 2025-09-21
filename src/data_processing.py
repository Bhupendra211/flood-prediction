import pandas as pd
import numpy as np


REQUIRED_DATE_COL= 'date'

def load_data(path:str)->pd.DataFrame:
    df = pd.read_csv(path)
    if REQUIRED_DATE_COL in df.columns:
        df[REQUIRED_DATE_COL] = pd.to_datetime(df[REQUIRED_DATE_COL], errors='coerce')
        df = df.sort_values(REQUIRED_DATE_COL).reset_index(drop=True)
        return df
    
def basic_eda(df: pd.DataFrame) -> dict:
    info = {
        'rows': df.shape[0],
        'cols': df.shape[1],
        'missing_count': df.isnull().sum().to_dict(),
        'date_min': df[REQUIRED_DATE_COL].min() if REQUIRED_DATE_COL in df.columns else None,
        'date_max': df[REQUIRED_DATE_COL].max() if REQUIRED_DATE_COL in df.columns else None,
        }
    return info


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if REQUIRED_DATE_COL not in df.columns:
        return df
    df['day'] = df[REQUIRED_DATE_COL].dt.day
    df['month'] = df[REQUIRED_DATE_COL].dt.month
    df['year'] = df[REQUIRED_DATE_COL].dt.year
    df['dayofweek'] = df[REQUIRED_DATE_COL].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5,6]).astype(int)
    # monsoon months (India): Jun-Sep
    df['is_monsoon'] = df['month'].isin([6,7,8,9]).astype(int)
    return df




def add_lag_and_rolling(df: pd.DataFrame, col: str='rainfall_mm', lags=[1,3,7], windows=[3,7,14]) -> pd.DataFrame:
    if col not in df.columns:
        print(f"Column {col} not in dataframe. Skipping lag/rolling creation.")
        return df


    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)


    for w in windows:
        df[f'{col}_roll_mean_{w}'] = df[col].rolling(window=w).mean()
        df[f'{col}_roll_sum_{w}'] = df[col].rolling(window=w).sum()


    # fill small number of NaNs with backward fill then forward
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df




def build_label_by_threshold(df: pd.DataFrame, level_col: str='river_level', threshold: float=None) -> pd.DataFrame:
    if level_col not in df.columns:
        raise ValueError(f"level_col '{level_col}' not present in dataframe")
    if threshold is None:
        threshold = df[level_col].quantile(0.90)
    df['flood_label'] = (df[level_col] >= threshold).astype(int)
    df['flood_threshold_used'] = float(threshold)
    return df




def get_last_month(df: pd.DataFrame) -> pd.DataFrame:
    if REQUIRED_DATE_COL not in df.columns:
        return df
    max_date = df[REQUIRED_DATE_COL].max()
    start = max_date - pd.Timedelta(days=30)
    return df[df[REQUIRED_DATE_COL] >= start].copy()