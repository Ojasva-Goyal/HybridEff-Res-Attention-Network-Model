import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_data(data_dir, batch_size=32, shuffle=True, transform=None):
    """
    Load data from the specified directory.

    Parameters:
    - data_dir (str): Path to the directory containing the data.
    - batch_size (int): Number of samples per batch.
    - shuffle (bool): Whether to shuffle the data.
    - transform (callable, optional): Optional transform to be applied on a sample.

    Returns:
    - DataLoader: PyTorch DataLoader for the dataset.
    """
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    if transform is None:
        transform = transforms.Compose([transforms.ToTensor()])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader

def save_model(model, path):
    """
    Save the trained model to the specified path.

    Parameters:
    - model (torch.nn.Module): Trained model to be saved.
    - path (str): Path to save the model.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load the trained model from the specified path.

    Parameters:
    - model (torch.nn.Module): Model architecture to load the state_dict into.
    - path (str): Path to load the model from.

    Returns:
    - model (torch.nn.Module): Model with loaded weights.
    """
    model.load_state_dict(torch.load(path))
    return model

def plot_loss(train_losses, val_losses, save_path=None):
    """
    Plot training and validation loss curves.

    Parameters:
    - train_losses (list of float): Training losses over epochs.
    - val_losses (list of float): Validation losses over epochs.
    - save_path (str, optional): Path to save the plot. If None, plot will be shown.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def set_seed(seed):
    """
    Set the random seed for reproducibility.

    Parameters:
    - seed (int): Seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_device():
    """
    Get the available device (CPU or GPU).

    Returns:
    - torch.device: Available device.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def accuracy(output, target):
    """
    Calculate the accuracy of the model.

    Parameters:
    - output (torch.Tensor): Model output.
    - target (torch.Tensor): Ground truth labels.

    Returns:
    - float: Accuracy score.
    """
    _, pred = torch.max(output, dim=1)
    correct = (pred == target).sum().item()
    return correct / len(target)
