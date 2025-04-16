import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.1):
        """
        Initialize the Classifier MLP
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dims (list): List of hidden layer dimensions
            num_classes (int): Number of output classes
            dropout (float): Dropout probability
        """
        super(ClassifierMLP, self).__init__()
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Combine all layers into a sequence
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Output logits with shape [batch_size, num_classes]
        """
        return self.mlp(x)
        
    def save_model(self, path):
        """
        Save the model
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
        
    def load_model(self, path):
        """
        Load the model
        
        Args:
            path (str): Path to the model file
        """
        self.load_state_dict(torch.load(path)) 