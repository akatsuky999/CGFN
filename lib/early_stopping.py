import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=8, delta=0.001, verbose=True):
        """
        Initializes the EarlyStopping object.

        Args:
            patience (int): The number of epochs to wait without improvement in the validation metric.
            delta (float): The minimum change in the validation metric to be considered as an improvement.
            verbose (bool): Whether to print messages when a new best model is saved or when the early stopping counter is updated.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_rmse_min = np.Inf

    def __call__(self, val_rmse, model, model_save_path):
        score = -val_rmse
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_rmse, model, model_save_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_rmse, model, model_save_path)
            self.counter = 0

    def save_checkpoint(self, val_rmse, model, model_save_path):
        """
        Saves the current best model when the validation RMSE improves.

        Args:
            val_rmse (float): The current validation root mean squared error.
            model (torch.nn.Module): The PyTorch model being trained.
            model_save_path (str): The path where the best model will be saved.
        """
        if self.verbose:
            print(f'Validation RMSE decreased ({self.val_rmse_min:.4f} --> {val_rmse:.4f}). Saving model...')
            print(" ")
        torch.save(model.state_dict(), model_save_path)
        self.val_rmse_min = val_rmse