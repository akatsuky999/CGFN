import os
import yaml
import torch
from numpy import load
from lib.utils import (
    prepare_spatiotemporal_data,
    generate_graph_matrices
)
from model.CGFN import CGFN
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def load_config(config_path):
    """Load and parse YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Normalize file paths
    for key in ['graph_signal_file', 'adj_file']:
        config['data'][key] = os.path.normpath(config['data'][key])

    return config


def calculate_metrics(pred, true):
    """Calculate RMSE, MAE, and MAPE"""
    mse = torch.nn.MSELoss()(pred, true)
    rmse = torch.sqrt(mse).item()
    mae = torch.nn.L1Loss()(pred, true).item()

    # Handle zero values for MAPE
    nonzero_mask = (true != 0)
    if nonzero_mask.sum().item() == 0:
        return rmse, mae, 0.0

    valid_true = true[nonzero_mask]
    valid_pred = pred[nonzero_mask]
    mape = torch.mean(torch.abs((valid_true - valid_pred) / valid_true)).item() * 100

    return rmse, mae, mape


def load_model_and_calculate_loss(config, model_path, start_percentage=10, end_percentage=20):
    data = load(config['data']['graph_signal_file'])
    num_nodes = int(data[data.files[0]].shape[1])

    adj_mx, _ = generate_graph_matrices(
        config['data']['adj_file'],
        num_nodes
    )
    _, _, val_loader, *_ = prepare_spatiotemporal_data(
        config['data']['dataset_id'],
        torch.device(config['train']['device']),
        config['train']['batch_size']
    )

    model = CGFN(
        DEVICE=torch.device(config['train']['device']),
        nb_block=config['model']['num_blocks'],
        in_channels=config['model']['in_channels'] + config['model']['num_embeddings'] * config['model'][
            'embedding_dim'],
        hidden_layer=config['model']['hidden_size'],
        hidden_time_layer=config['model']['time_hidden_size'],
        time_strides=config['model']['time_strides'],
        adj_mx=adj_mx,
        num_for_predict=config['train']['pred_steps'],
        len_input=config['train']['seq_len'],
        num_of_vertices=num_nodes,
        emb=config['model']['embedding_dim']
    ).to(config['train']['device'])

    state_dict = torch.load(model_path, weights_only=True)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)
    model.eval()

    all_true = []
    all_pred = []
    val_metrics = {'RMSE': 0.0, 'MAE': 0.0, 'MAPE': 0.0}
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(config['train']['device'])
            targets = targets.to(config['train']['device'])
            outputs = model(inputs)
            rmse, mae, mape = calculate_metrics(outputs, targets)
            val_metrics['RMSE'] += rmse * inputs.size(0)
            val_metrics['MAE'] += mae * inputs.size(0)
            val_metrics['MAPE'] += mape * inputs.size(0)

            all_true.extend(targets.cpu().numpy().flatten())
            all_pred.extend(outputs.cpu().numpy().flatten())

    val_rmse = val_metrics['RMSE'] / len(val_loader.dataset)
    val_mae = val_metrics['MAE'] / len(val_loader.dataset)
    val_mape = val_metrics['MAPE'] / len(val_loader.dataset)

    start_index = int(len(all_true) * start_percentage / 100)
    end_index = int(len(all_true) * end_percentage / 100)
    sliced_true = all_true[start_index:end_index]
    sliced_pred = all_pred[start_index:end_index]

    plt.figure(figsize=(12, 6))
    plt.plot(sliced_true, label='True Values', color='blue', linewidth=1.5)
    plt.plot(sliced_pred, label='Predicted Values', color='orange', linewidth=1.5)
    plt.xlabel('Sample Index', fontweight='bold')
    plt.ylabel('Value', fontweight='bold')
    plt.title('True vs Predicted Values', fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('predicted.png')
    plt.show()

    return val_rmse, val_mae, val_mape


if __name__ == "__main__":
    config = load_config("Data/PEMS04/config.yaml")
    model_path = os.path.join(config['data']['save_dir'], "BEST_CGFN_202504091133.pth")
    rmse, mae, mape = load_model_and_calculate_loss(config, model_path, start_percentage=3, end_percentage=3.0025)
    print(f"Validation RMSE: {rmse:.4f}")
    print(f"Validation MAE: {mae:.4f}")
    print(f"Validation MAPE: {mape:.2f}%")
