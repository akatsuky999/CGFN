import os
import torch
import torch.utils.data
import csv
import numpy as np

def generate_graph_matrices(data_file, node_count, node_map=None):
    '''
    Create adjacency and distance matrices from spatial data

    Parameters:
    data_file (str): Path to edge data file (CSV/NPY)
    node_count (int): Total number of graph nodes
    node_map (str): Optional file for node ID mapping

    Returns:
    tuple: (adjacency_matrix, distance_matrix)
    '''
    if data_file.endswith('.npy'):
        return np.load(data_file), None

    adj_matrix = np.zeros((node_count, node_count), dtype=np.float32)
    dist_matrix = np.zeros_like(adj_matrix)

    id_converter = {}
    if node_map:
        with open(node_map) as f:
            id_converter = {int(nid): idx for idx, nid in enumerate(f.read().splitlines())}

    with open(data_file) as f:
        csv.reader(f).__next__()  # Skip header
        for record in csv.reader(f):
            if len(record) != 3:
                continue

            src, dest, weight = int(record[0]), int(record[1]), float(record[2])
            if node_map:
                src, dest = id_converter[src], id_converter[dest]

            adj_matrix[src, dest] = 1.0
            dist_matrix[src, dest] = weight

    return adj_matrix, dist_matrix


def prepare_spatiotemporal_data(filename, DEVICE, batch_size, shuffle=True, idx: int = 0):
    """
        Prepare spatiotemporal data for training, validation, and testing.

        This function loads a `.npz` file containing spatiotemporal data, selects specific features,
        converts the data into PyTorch tensors, creates datasets and data loaders, and finally
        prints a summary of the data shapes.

        Args:
            filename (str): The name of the `.npz` file (without the extension) containing the data.
            DEVICE (torch.device): The device (CPU or GPU) where the tensors will be stored.
            batch_size (int): The number of samples per batch to load.
            shuffle (bool, optional): Whether to shuffle the data at every epoch. Defaults to True.
            idx (int, optional): The starting index for feature selection. Defaults to 0.

        Returns:
            tuple: A tuple containing the following elements:
                - train_loader (torch.utils.data.DataLoader): Data loader for the training set.
                - train_target_tensor (torch.Tensor): Target tensor for the training set.
                - val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
                - val_target_tensor (torch.Tensor): Target tensor for the validation set.
                - test_loader (torch.utils.data.DataLoader): Data loader for the test set.
                - test_target_tensor (torch.Tensor): Target tensor for the test set.
                - mean (np.ndarray): Mean values of the selected features.
                - std (np.ndarray): Standard deviation values of the selected features.
        """

    folder_path = 'train_data'
    file_name = filename
    full_path = os.path.join(folder_path, file_name)
    print('load file:', full_path)

    file_data = np.load(full_path + '.npz')

    def select_features(data):
        return data[:, :, [idx, idx + 3, idx + 4], :]

    train_x = select_features(file_data['train_x'])
    train_target = file_data['train_target']
    val_x = select_features(file_data['val_x'])
    val_target = file_data['val_target']
    test_x = select_features(file_data['test_x'])
    test_target = file_data['test_target']
    mean = select_features(file_data['mean'])
    std = select_features(file_data['std'])

    def create_tensor(data, device):
        return torch.from_numpy(data).type(torch.FloatTensor).to(device)

    def create_loader(x, target, batch_size, shuffle_flag, device):
        x_tensor = create_tensor(x, device)
        target_tensor = create_tensor(target, device)
        dataset = torch.utils.data.TensorDataset(x_tensor, target_tensor)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=shuffle_flag), x_tensor, target_tensor

    train_loader, train_x_tensor, train_target_tensor = create_loader(train_x, train_target, batch_size, shuffle,
                                                                      DEVICE)
    val_loader, val_x_tensor, val_target_tensor = create_loader(val_x, val_target, batch_size, False, DEVICE)
    test_loader, test_x_tensor, test_target_tensor = create_loader(test_x, test_target, batch_size, False, DEVICE)

    print("-" * 93)
    print("Data Shapes Summary:")
    print("-" * 93)
    print(f"Train Data:      X -> {train_x_tensor.size()}, Target -> {train_target_tensor.size()}")
    print(f"Validation Data: X -> {val_x_tensor.size()}, Target -> {val_target_tensor.size()}")
    print(f"Test Data:       X -> {test_x_tensor.size()}, Target -> {test_target_tensor.size()}")
    print("-" * 93)
    print("Data Loaders Created Successfully!")
    print("=" * 93)

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std

