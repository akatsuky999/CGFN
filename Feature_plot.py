import numpy as np
import matplotlib.pyplot as plt

# Initialize plotting style and parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 14,
    'axes.linewidth': 1.5,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2,
    'savefig.dpi': 300
})


def load_and_prepare_data(data_path):
    """Load and preprocess time series data from NPZ file.

    Args:
        data_path: Path to the NPZ data file

    Returns:
        Tuple of (train_x, train_timestamp) where:
        train_x: Time series data with shape (B, N, F, T)
        train_timestamp: Corresponding timestamps
    """
    data = np.load(data_path)
    train_x = data['train_x']
    train_timestamp = data['train_timestamp'].squeeze() if 'train_timestamp' in data else np.arange(train_x.shape[0])
    return train_x, train_timestamp


def extract_time_series(data, timestamps, node_idx, start_t, num_points):
    """Extract time series segment for visualization.

    Args:
        data: Full time series data (B, N, F, T)
        timestamps: Corresponding time indices
        node_idx: Node index to visualize
        start_t: Starting time index
        num_points: Number of consecutive points to plot

    Returns:
        Tuple of (time_series, time_axis) where:
        time_series: Extracted features (N, F)
        time_axis: Corresponding time values
    """
    time_series = data[start_t:start_t + num_points, node_idx, :, 0]
    time_axis = timestamps[start_t:start_t + num_points]
    return time_series, time_axis


def plot_temporal_evolution(time_series, time_axis, node_idx, num_points):
    """Generate temporal evolution plot of features.

    Args:
        time_series: Feature data to plot (N, F)
        time_axis: X-axis time values
        node_idx: Node identifier for title
        num_points: Number of points for title
    """
    plt.figure()
    num_features = time_series.shape[1]

    for i in range(num_features):
        plt.plot(time_axis, time_series[:, i], label=f'Feature {i + 1}')

    plt.xlabel('Time')
    plt.ylabel('Normalized Value')
    plt.title(f'Temporal Features (Node {node_idx}, {num_points} points)')
    plt.legend(loc='best', frameon=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Configuration parameters
    DATA_PATH = r'train_data\PEMS04_w0_d0_h1.npz'
    NODE_INDEX = 0
    START_TIME = 100
    NUM_POINTS = 500

    # Data processing pipeline
    train_data, timestamps = load_and_prepare_data(DATA_PATH)
    ts_data, ts_axis = extract_time_series(train_data, timestamps,
                                           NODE_INDEX, START_TIME, NUM_POINTS)
    plot_temporal_evolution(ts_data, ts_axis, NODE_INDEX, NUM_POINTS)