import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Load the NASA Turbofan dataset.
    """
    columns = ["engine_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + \
              [f"sensor_{i}" for i in range(1, 22)]
    return pd.read_csv(filepath, sep="\\s+", header=None, names=columns)

def calculate_rul(df):
    """
    Calculate the Remaining Useful Life (RUL) for each row.
    RUL is defined as the max cycle minus the current cycle for an engine.
    """
    df["RUL"] = df.groupby("engine_id")["cycle"].transform("max") - df["cycle"]
    return df

def feature_engineering(df, window=5):
    """
    Create rolling mean and standard deviation features for sensor readings.
    """
    df_feat = df.copy()
    sensor_cols = [c for c in df.columns if "sensor" in c]
    
    for col in sensor_cols:
        df_feat[f"{col}_roll_mean"] = df_feat.groupby("engine_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df_feat[f"{col}_roll_std"] = df_feat.groupby("engine_id")[col].transform(
            lambda x: x.rolling(window, min_periods=1).std().fillna(0)
        )
    return df_feat

def remove_constant_sensors(df):
    """
    Remove sensors that have near-zero variance.
    """
    variances = df.var(numeric_only=True)
    useless_sensors = [col for col in variances.index if variances[col] < 1e-10]
    return df.drop(columns=useless_sensors)

def prepare_data(filepath, window=5):
    """
    End-to-end preprocessing pipeline.
    """
    df = load_data(filepath)
    df = calculate_rul(df)
    df = remove_constant_sensors(df)
    df = feature_engineering(df, window=window)
    return df
