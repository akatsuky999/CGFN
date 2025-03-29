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


# ------------------------------ Execution Flow ------------------------------
if __name__ == "__main__":
    CONFIG = {
        'data_path': 'Data/PEMS04/PEMS04.npz',
        'nodes': 307,
        'points_per_hour': 12,
        'input_steps': 12,         # Number of historical time steps for input
        'pred_steps': 3,           # Number of prediction time steps
        'temporal_windows': {'weeks': 0, 'days': 0, 'hours': 1}
    }

    # ==================== DATA PROCESSING ====================
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 20 + "DATA PROCESSING STARTED" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")

    print(f"\n{' CONFIGURATION ':-^60}")
    print(f"| {'Parameter':<25} | {'Value':<30} |")
    print(f"| {'-' * 25} | {'-' * 30} |")
    print(f"| {'Data path':<25} | {CONFIG['data_path']:<30} |")
    print(f"| {'Number of nodes':<25} | {CONFIG['nodes']:<30} |")
    print(f"| {'Points per hour':<25} | {CONFIG['points_per_hour']:<30} |")
    print(f"| {'Input steps':<25} | {CONFIG['input_steps']:<30} |")
    print(f"| {'Prediction steps':<25} | {CONFIG['pred_steps']:<30} |")
    print(f"| {'Temporal windows':<25} | {str(CONFIG['temporal_windows']):<30} |")
    print("-" * 60)

    # Process dataset
    train_set, val_set, test_set = process_dataset(
        CONFIG['data_path'],
        CONFIG['temporal_windows']['weeks'],
        CONFIG['temporal_windows']['days'],
        CONFIG['temporal_windows']['hours'],
        CONFIG['input_steps'],
        CONFIG['pred_steps'],
        CONFIG['points_per_hour']
    )

    # ==================== DATA DIMENSIONS ====================
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 21 + "DATA DIMENSIONS" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")

    train_x = np.concatenate(train_set[:-2], axis=-1)
    val_x = np.concatenate(val_set[:-2], axis=-1)
    test_x = np.concatenate(test_set[:-2], axis=-1)

    print(f"\n{' DATASET SHAPES ':-^60}")
    print(f"| {'Dataset':<15} | {'Features':<20} | {'Targets':<20} |")
    print(f"| {'-' * 15} | {'-' * 20} | {'-' * 20} |")
    print(f"| {'Training':<15} | {str(train_x.shape):<20} | {str(train_set[-2].shape):<20} |")
    print(f"| {'Validation':<15} | {str(val_x.shape):<20} | {str(val_set[-2].shape):<20} |")
    print(f"| {'Testing':<15} | {str(test_x.shape):<20} | {str(test_set[-2].shape):<20} |")
    print("-" * 60)

    # Normalize data
    stats, norm_train, norm_val, norm_test = normalize_data(
        train_x, val_x, test_x
    )

    # ==================== STATISTICS ====================
    print(f"\n{' NORMALIZATION STATISTICS ':-^60}")
    print(f"| {'Statistic':<25} | {'Shape':<30} |")
    print(f"| {'-' * 25} | {'-' * 30} |")
    print(f"| {'Mean':<25} | {str(stats['mean'].shape):<30} |")
    print(f"| {'Standard deviation':<25} | {str(stats['std'].shape):<30} |")
    print("-" * 60)

    # Prepare save path
    save_dir = 'train_data'
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"TTPEMS04_w{CONFIG['temporal_windows']['weeks']}_d{CONFIG['temporal_windows']['days']}_h{CONFIG['temporal_windows']['hours']}_I{CONFIG['input_steps']}_P{CONFIG['pred_steps']}"
    save_path = os.path.join(save_dir, save_name)

    # ==================== SAVING DATA ====================
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 22 + "SAVING DATA" + " " * 25 + "║")
    print("╚" + "═" * 58 + "╝")

    print(f"\n{' SAVE DETAILS ':-^60}")
    print(f"| {'Directory':<25} | {save_dir:<30} |")
    print(f"| {'File name':<25} | {save_name:<30} |")
    print(f"| {'Full path':<25} | {save_path:<30} |")
    print("-" * 60)

    # Save processed data
    np.savez_compressed(
        save_path,
        train_x=norm_train,
        train_target=train_set[-2],
        val_x=norm_val,
        val_target=val_set[-2],
        test_x=norm_test,
        test_target=test_set[-2],
        mean=stats['mean'],
        std=stats['std']
    )

    # ==================== COMPLETION ====================
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " " * 18 + "PROCESSING COMPLETED" + " " * 20 + "║")
    print("╚" + "═" * 58 + "╝")
    print(f"\n{' Final output saved to: ':-^60}")
    print(f"{save_path}")
    print("-" * 60 + "\n")
