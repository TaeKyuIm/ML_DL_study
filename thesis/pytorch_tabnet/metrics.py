from dataclasses import dataclass
from typing import List
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    log_loss,
    balanced_accuracy_score,
    mean_squared_log_error
)
import torch

def UnsupervisedLoss(y_pred, embedded_x, obf_vars, eps=1e-9):
    """
    Implements unsupervised loss function.
    This differs from orginal paper as it's scaled to be batch size independent
    and number of features reconstructed independent (by taking the mean)
    Parameters
    ----------
    y_pred : torch.Tensor or np.array
        Reconstructed prediction (with embeddings)
    embedded_x : torch.Tensor
        Original input embedded by network
    obf_vars : torch.Tensor
        Binary mask for obfuscated variables.
        1 means the variable was obfuscated so reconstruction is based on this.
    eps : float
        A small floating point to avoid ZeroDivisionError
        This can happen in degenerated case when a feature has only one value
    Returns
    -------
    loss : torch float
        Unsupervised loss, average value over batch samples.
    """
    errors = y_pred - embedded_x
    reconstruction_errors = torch.mul(errors, obf_vars) ** 2
    batch_means = torch.mean(embedded_x, dim=0)
    batch_means[batch_means == 0] = 1
    
    batch_stds = torch.std(embedded_x, dim=0) ** 2
    batch_stds[batch_stds == 0] = batch_means[batch_stds == 0]
    feature_loss = torch.matmul(reconstruction_errors, 1 / batch_stds)
    
    nb_reconstructed_variables = torch.sum(obf_vars, dim=1)
    
    feature_loss = feature_loss / (nb_reconstructed_variables + eps)
    
    loss = torch.mean(feature_loss)
    return loss

def UnsupervisedLossNumpy(y_pred, embedded_x, obf_vars, eps=1e-9):
    errors = y_pred - embedded_x
    reconstruction_errors = np.multiply(errors, obf_vars)**2
    batch_means = np.mean(embedded_x, axis=0)
    batch_means = np.where(batch_means == 0, 1, batch_means)
    
    batch_stds = np.std(embedded_x, axis=0, ddof=1)**2
    batch_stds = np.where(batch_stds == 0, batch_means, batch_stds)
    feature_loss = np.matmul(reconstruction_errors, 1 / batch_stds)
    
    nb_reconstructed_variables = np.sum(obf_vars, axis=1)
    
    feature_loss = feature_loss / (nb_reconstructed_variables + eps)
    
    loss = np.mean(feature_loss)
    return loss
class Metric:
    def __call__(self, y_true, y_pred):
        raise NotImplementedError("Custome Metrics must implement this function")
    
    @classmethod
    def get_metrics_by_name(cls, names):
        """Get list of metric classes.
        Parameters
        ----------
        cls : Metric
            Metric class.
        names : list
            List of metric names.
        Returns
        -------
        metrics : list
            List of metric classes.
        """
        available_metrics = cls.__subclasses__()
        available_names = [metric()._name for metric in available_metrics]
        metrics = []
        for name in names:
            assert (
                name in available_names
            ), f"{name} is not available, choose in {available_names}"
            idx = available_names.index(name)
            metric = available_metrics[idx]()
            metrics.append(metric)
        return metrics
@dataclass
class UnsupMetricContainer:
    metric_names: List[str]
    prefix: str = ""
    
    def __post_init__(self):
        self.metrics = Metric.get_metrics_by_name(self.metric_names)
        self.names = [self.prefix + name for name in self.metric_names]
        
    def __call__(self, y_pred, embedded_x, obf_vars):
        logs = {}
        for metric in self.metrics:
            res = metric(y_pred, embedded_x, obf_vars)
            logs[self.prefix + metric._name] = res
        return logs