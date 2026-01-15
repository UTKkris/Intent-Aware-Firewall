

import pandas as pd
import numpy as np

def add_time_features(df):
    
    df = df.copy()
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    return df

def is_suspicious_time(hour, suspicious_hours=None):
    
    if suspicious_hours is None:
        suspicious_hours = set(range(0, 6))
    return hour in suspicious_hours

def add_suspicious_time_flag(df, suspicious_hours=None):
    
    if 'Hour' not in df.columns:
        df = add_time_features(df)
    df = df.copy()
    df['SuspiciousTime'] = df['Hour'].apply(
        lambda h: 1 if is_suspicious_time(h, suspicious_hours) else 0
    )
    return df

def add_packet_size_flags(df, size_col='Packet Length', mean=None, std=None):
    
    if size_col not in df.columns:
        df['SizeAnomaly'] = 0
        return df, 0, 0
    df = df.copy()
    if mean is None:
        mean = df[size_col].mean()
    if std is None:
        std = df[size_col].std()
    if std == 0 or pd.isna(std):
        df['SizeAnomaly'] = 0
        return df, mean, 0
    threshold = 3 * std
    df['SizeAnomaly'] = df[size_col].apply(
        lambda x: 1 if abs(x - mean) > threshold else 0
    )
    return df, mean, std

def calculate_windowed_features(df, window='1S', rolling_window=3):
    
    if 'Src IP' not in df.columns or 'Timestamp' not in df.columns:
        return pd.DataFrame()
    df_indexed = df.set_index('Timestamp')
    packet_counts_obj = df_indexed.groupby('Src IP').resample(window).size()
    if isinstance(packet_counts_obj, pd.Series):
        packet_counts = packet_counts_obj.to_frame(name='PacketRate')
    else:
        packet_counts = packet_counts_obj
        count_col = packet_counts.columns[-1]
        packet_counts = packet_counts.rename(columns={count_col: 'PacketRate'})
        if 'Src IP' in packet_counts.columns:
            packet_counts = packet_counts.drop(columns='Src IP')
    burst_feature_series = packet_counts.groupby(level=0)['PacketRate'] \
                                        .rolling(window=rolling_window) \
                                        .std()
    windowed_features = packet_counts
    windowed_features['BurstFeature'] = burst_feature_series.values
    windowed_features = windowed_features.fillna(0)
    if 'Timestamp' in windowed_features.columns:
        windowed_features = windowed_features.drop(columns='Timestamp')
    return windowed_features.reset_index()
