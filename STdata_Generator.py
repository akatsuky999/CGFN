import os
import numpy as np
import pandas as pd


# ------------------------------ Core Data Processing ------------------------------
def search_data(sequence_length, num_depend, label_start, window_steps, time_units, points_per_hour):
    """Generate sliding window indices for temporal data sampling.

    Args:
        sequence_length: Total length of time sequence
        num_depend: Number of historical windows needed
        label_start: Start index for prediction
        window_steps: Window length for historical data
        time_units: Time resolution units (7*24 for weekly, 24 for daily)
        points_per_hour: Data points per hour

    Returns:
        List of reversed (start, end) index tuples for historical windows
    """
    if points_per_hour < 1:
        raise ValueError("Invalid points_per_hour value")

    if label_start + window_steps > sequence_length:
        return None

    indices = []
    for i in range(1, num_depend + 1):
        start = label_start - points_per_hour * time_units * i
        end = start + window_steps

        if start >= 0:
            indices.append((start, end))
        else:
            return None

    return indices[::-1]


def get_sample_indices(data_seq, num_weeks, num_days, num_hours, label_start, input_steps, pred_steps, points_per_hour=12):
    """Extract multi-scale temporal features from time series data.

    Returns:
        week_data: Weekly patterns (samples, nodes, features)
        day_data: Daily patterns (samples, nodes, features)
        hour_data: Recent patterns (samples, nodes, features)
        target: Prediction targets (steps, nodes, features)
    """
    if label_start + pred_steps > data_seq.shape[0]:
        return None, None, None, None

    week_data, day_data, hour_data = None, None, None

    if num_weeks > 0:
        week_indices = search_data(data_seq.shape[0], num_weeks, label_start,
                                   input_steps, 7 * 24, points_per_hour)
        if week_indices:
            week_data = np.concatenate([data_seq[i:j] for i, j in week_indices], axis=0)

    if num_days > 0:
        day_indices = search_data(data_seq.shape[0], num_days, label_start,
                                  input_steps, 24, points_per_hour)
        if day_indices:
            day_data = np.concatenate([data_seq[i:j] for i, j in day_indices], axis=0)

    if num_hours > 0:
        hour_indices = search_data(data_seq.shape[0], num_hours, label_start,
                                   input_steps, 1, points_per_hour)
        if hour_indices:
            hour_data = np.concatenate([data_seq[i:j] for i, j in hour_indices], axis=0)

    target = data_seq[label_start: label_start + pred_steps]
    return week_data, day_data, hour_data, target


def add_time_features(data, daily_points):
    """Enrich data with temporal position encodings.

    Args:
        data: Raw input (seq_len, nodes, features)
        daily_points: Data points per day

    Returns:
        Enhanced data with time features (seq_len, nodes, features+2)
    """
    seq_len, nodes, _ = data.shape
    features = [data]

    time_feature = np.array([i % daily_points / daily_points for i in range(seq_len)])
    features.append(np.tile(time_feature, [nodes, 1]).T[:, :, None])

    day_feature = np.array([(i // daily_points) % 7 / 7 for i in range(seq_len)])
    features.append(np.tile(day_feature, [nodes, 1]).T[:, :, None])

    return np.concatenate(features, axis=-1)


def process_dataset(data_file, num_weeks, num_days, num_hours, input_steps, pred_steps, points_per_hour=12):
    """Main pipeline for generating time-series dataset."""
    raw_data = np.load(data_file)['data']
    daily_points = points_per_hour * 24
    data = add_time_features(raw_data, daily_points)

    samples = []
    for idx in range(data.shape[0]):
        week, day, hour, target = get_sample_indices(
            data, num_weeks, num_days, num_hours, idx, input_steps, pred_steps, points_per_hour
        )

        # ----------------------------------------------------------------
        is_valid = True
        if num_weeks > 0:
            if week is None:
                is_valid = False
        if num_days > 0:
            if day is None:
                is_valid = False
        if num_hours > 0:
            if hour is None:
                is_valid = False

        if not is_valid:
            continue

        # ---------------------------------------------------------------
        components = []
        
        if num_weeks > 0:
            week_processed = week[np.newaxis].transpose(0, 2, 3, 1)  # (1,N,F,T)
            components.append(week_processed)
        
        if num_days > 0:
            day_processed = day[np.newaxis].transpose(0, 2, 3, 1)
            components.append(day_processed)
        
        if num_hours > 0:
            hour_processed = hour[np.newaxis].transpose(0, 2, 3, 1)
            components.append(hour_processed)

        target_processed = target[np.newaxis].transpose(0, 2, 3, 1)[:, :, 0, :]  # (1,N,T)
        components.extend([target_processed, np.array([[idx]])])

        samples.append(components)

    # ----------------------------------------------------------------
    split1 = int(len(samples) * 0.6)
    split2 = int(len(samples) * 0.8)

    train = [np.concatenate(c) for c in zip(*samples[:split1])]
    val = [np.concatenate(c) for c in zip(*samples[split1:split2])]
    test = [np.concatenate(c) for c in zip(*samples[split2:])]

    return train, val, test


def normalize_data(train, val, test):
    """Apply Z-score normalization using training statistics.

    Returns:
        stats: Dictionary with mean/std values
        Normalized train/val/test sets
    """
    assert train.shape[1:] == val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=(0, 1, 3), keepdims=True)
    std = train.std(axis=(0, 1, 3), keepdims=True)

    def zscore(x):
        return (x - mean) / std

    return {'mean': mean, 'std': std}, zscore(train), zscore(val), zscore(test)



