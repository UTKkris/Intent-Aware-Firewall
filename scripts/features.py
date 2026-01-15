"""
Feature Engineering for Network Traffic Analysis.

This module provides functions to extract various features from a network
traffic DataFrame (assumed to be a pandas DataFrame). These features are
designed to help machine learning models detect anomalies, such as those
from botnets or other malicious activity.

Functions:
- add_time_features: Extracts hour and day of the week from timestamps.
- is_suspicious_time: Helper function to check if an hour is "off-hours".
- add_suspicious_time_flag: Adds a binary flag for suspicious time.
- add_packet_size_flags: Adds a binary flag for anomalous packet sizes.
- calculate_windowed_features: Calculates stateful features like packet rate
                               and burstiness over time windows.
"""

import pandas as pd
import numpy as np

def add_time_features(df):
    """
    Extracts temporal features (Hour, DayOfWeek) from the 'Timestamp' column.

    Args:
        df (pd.DataFrame): DataFrame containing a 'Timestamp' column.

    Returns:
        pd.DataFrame: The DataFrame with 'Hour' and 'DayOfWeek' columns added.
    """
    df = df.copy()
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    return df

def is_suspicious_time(hour, suspicious_hours=None):
    """
    Checks if a given hour falls within a set of suspicious hours.

    Args:
        hour (int): The hour of the day (0-23).
        suspicious_hours (set, optional): A set of integers representing
            suspicious hours. Defaults to {0, 1, 2, 3, 4, 5}.

    Returns:
        bool: True if the hour is in the suspicious set, False otherwise.
    """
    if suspicious_hours is None:
        suspicious_hours = set(range(0, 6))
    return hour in suspicious_hours

def add_suspicious_time_flag(df, suspicious_hours=None):
    """
    Adds a binary 'SuspiciousTime' flag to the DataFrame.

    If the 'Hour' column does not exist, it calls `add_time_features` first.

    Args:
        df (pd.DataFrame): DataFrame with an 'Hour' column.
        suspicious_hours (set, optional): Passed to `is_suspicious_time`.
            Defaults to {0, 1, 2, 3, 4, 5}.

    Returns:
        pd.DataFrame: The DataFrame with the 'SuspiciousTime' column added.
    """
    if 'Hour' not in df.columns:
        df = add_time_features(df)
    df = df.copy()
    df['SuspiciousTime'] = df['Hour'].apply(
        lambda h: 1 if is_suspicious_time(h, suspicious_hours) else 0
    )
    return df

def add_packet_size_flags(df, size_col='Packet Length', mean=None, std=None):
    """
    Adds a binary flag for anomalous packet sizes based on standard deviation.

    This function can operate in two modes:
    1.  Fit & Transform (mean/std are None): Calculates mean/std from `df`
        and adds the 'SizeAnomaly' column.
    2.  Transform (mean/std are provided): Uses the given mean/std (e.g., from
        a training set) to add the 'SizeAnomaly' column to `df`.

    Args:
        df (pd.DataFrame): DataFrame with a packet size column.
        size_col (str, optional): The name of the packet size column.
            Defaults to 'Packet Length'.
        mean (float, optional): Pre-calculated mean of the size column.
        std (float, optional): Pre-calculated standard deviation of the size column.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The DataFrame with 'SizeAnomaly' column.
            - float: The mean used for the calculation (calculated or passed in).
            - float: The std dev used for the calculation (calculated or passed in).
    """
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
    """
    Calculates windowed packet rate and burst features grouped by 'Src IP'.
    """
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
