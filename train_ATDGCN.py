import os
import yaml
import time
import torch
import datetime
from torch import optim
from model.TAD_Net import TAD_Net
from numpy import load
from lib.utils import (
    prepare_spatiotemporal_data,
    generate_graph_matrices
)
from lib.early_stopping import EarlyStopping

def load_config(config_path):
    """Load and parse YAML configuration"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Normalize file paths
    for key in ['graph_signal_file', 'adj_file']:
        config['data'][key] = os.path.normpath(config['data'][key])
        
    return config

def setup_logging(config):
    """Initialize logging directories"""
    os.makedirs(config['data']['log_dir'], exist_ok=True)
    os.makedirs(config['data']['save_dir'], exist_ok=True)
    
    now = datetime.datetime.now()
    log_filename = f"ATDGCN_{now.strftime('%Y%m%d%H%M')}.log"
    best_model_name = f"BEST_ATDGCN_{now.strftime('%Y%m%d%H%M')}.pth"
    best_model_path = os.path.join(config['data']['save_dir'], best_model_name)
    log_filepath = os.path.join(config['data']['log_dir'], log_filename)
    best_model_path = os.path.join(config['data']['save_dir'], best_model_name)
    
    return log_filepath , best_model_path

def print_config(config, log_file, best_model_path):
    """Print training configuration"""
    config_str = "="*50 + "\nTraining Configuration:\n"
    config_str += yaml.dump(config, default_flow_style=False)
    config_str += f"Best Model Path: {best_model_path}\n" + "="*50
    
    print(config_str)
    with open(log_file, 'w') as f:
        f.write(config_str)

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
    mape = torch.mean(torch.abs((valid_true - valid_pred)/valid_true)).item()*100
    
    return rmse, mae, mape

def main(_Config):
    config = load_config(_Config)
    log_filepath, best_model_path = setup_logging(config)
    print_config(config, log_filepath, best_model_path)
    
    # Load spatial data
    data = load(config['data']['graph_signal_file'])
    num_nodes = int(data[data.files[0]].shape[1])
    # num_nodes = config['data']['num_of_node']
    
    # Generate adjacency matrices
    adj_mx, _ = generate_graph_matrices(
        config['data']['adj_file'],
        num_nodes
    )
    
    # Prepare data loaders
    train_loader, _, val_loader, *_ = prepare_spatiotemporal_data(
        config['data']['dataset_id'],
        torch.device(config['train']['device']),
        config['train']['batch_size']
    )
    
    # Initialize model
    model = TAD_Net(
        DEVICE=torch.device(config['train']['device']),
        nb_block=config['model']['num_blocks'],
        in_channels=config['model']['in_channels'] + config['model']['num_embeddings']*config['model']['embedding_dim'],
        hidden_layer=config['model']['hidden_size'],
        hidden_time_layer=config['model']['time_hidden_size'],
        time_strides=config['model']['time_strides'],
        adj_mx=adj_mx,
        num_for_predict=config['train']['pred_steps'],
        len_input=config['train']['seq_len'],
        num_of_vertices=num_nodes,
        emb=config['model']['embedding_dim']
    ).to(config['train']['device'])

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters - Total: {total_params:,} | Trainable: {trainable_params:,}\n")

    # Configure optimizer (dynamic learning rate adjustment removed)
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'])
    
    early_stopping = EarlyStopping(
        patience=config['train']['patience'],
        delta=config['train']['delta']
    )
    
    total_time = 0
    for epoch in range(config['train']['epochs']):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_metrics = {'RMSE': 0.0, 'MAE': 0.0, 'MAPE': 0.0}
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = torch.sqrt(torch.nn.MSELoss()(outputs, targets))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['grad_clip'])
            optimizer.step()
            
            rmse, mae, mape = calculate_metrics(outputs, targets)
            train_metrics['RMSE'] += rmse * inputs.size(0)
            train_metrics['MAE'] += mae * inputs.size(0)
            train_metrics['MAPE'] += mape * inputs.size(0)
        
        # Validation phase
        model.eval()
        val_metrics = {'RMSE': 0.0, 'MAE': 0.0, 'MAPE': 0.0}
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                rmse, mae, mape = calculate_metrics(outputs, targets)
                val_metrics['RMSE'] += rmse * inputs.size(0)
                val_metrics['MAE'] += mae * inputs.size(0)
                val_metrics['MAPE'] += mape * inputs.size(0)
        
        # Compute metrics (learning rate remains constant)
        train_rmse = train_metrics['RMSE'] / len(train_loader.dataset)
        train_mae = train_metrics['MAE'] / len(train_loader.dataset)
        train_mape = train_metrics['MAPE'] / len(train_loader.dataset)
        val_rmse = val_metrics['RMSE'] / len(val_loader.dataset)
        val_mae = val_metrics['MAE'] / len(val_loader.dataset)
        val_mape = val_metrics['MAPE'] / len(val_loader.dataset)
        prev_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch details
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start
        total_time += epoch_time
        log_str = (
            f"Epoch {epoch+1:03}/{config['train']['epochs']} | "
            f"Time: {epoch_time:.1f}s | Total: {total_time:.1f}s | LR: {current_lr:.2e}\n"
            f"Train RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | MAPE: {train_mape:.2f}%\n"
            f"Val   RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f} | MAPE: {val_mape:.2f}%\n"
        )
        
        if current_lr != prev_lr:
            log_str += f"LR changed: {prev_lr:.2e} → {current_lr:.2e}\n"
        
        print(log_str)
        with open(log_filepath, 'a') as f:
            f.write(log_str)
        
        # Early stopping check
        early_stopping(val_rmse, model, best_model_path)
        if early_stopping.early_stop:
            print(f"Early stop at epoch {epoch+1}")
            print(f"Best Validation RMSE: {early_stopping.val_rmse_min:.4f}, model saved at: {best_model_path}")
            break
    
    # Save final model
    final_path = os.path.join(config['data']['save_dir'], "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"Training completed in {total_time//3600:.0f}h {total_time%3600//60:.0f}m")

if __name__ == "__main__":
    main("Data/PEMS04/config.yaml")
