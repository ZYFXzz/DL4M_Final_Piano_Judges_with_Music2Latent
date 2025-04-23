import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the model architecture"""
    num_encoder_layers: int = 2
    dim_transformer: int = 128
    dim_feedforward: int = 64
    dropout: float = 0.1
    nhead: int = 2

@dataclass
class TrainingConfig:
    """Configuration for training"""
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_classes: int = 7
    n_segs: int = 10

class AudioCNNTransformer(nn.Module):
    def __init__(self, h: int, w: int, n_segs: int, out_classes: int, 
                 config: ModelConfig):
        """
        Audio CNN-Transformer model
        
        Args:
            h (int): Height of input feature
            w (int): Width of input feature
            n_segs (int): Number of segments
            out_classes (int): Number of output classes
            config (ModelConfig): Model configuration
        """
        super(AudioCNNTransformer, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        
        # Projection layer to match transformer dimension
        self.projection = nn.Linear(128 * (h//8) * (w//8), config.dim_transformer)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_transformer,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Output layer
        self.fc = nn.Linear(config.dim_transformer, out_classes)
        
    def forward(self, x):
        # CNN
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Reshape and project to transformer dimension
        batch_size = x.size(0)
        x = x.view(batch_size, -1)  # Flatten
        x = self.projection(x)  # Project to transformer dimension
        x = x.view(batch_size, -1, x.size(-1))  # Reshape for transformer
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output
        x = self.fc(x)
        return x

class SingleLabelClassifier(pl.LightningModule):
    def __init__(self, embedding_dim: int, embedding_len: int, 
                 model_config: Optional[ModelConfig] = None,
                 training_config: Optional[TrainingConfig] = None):
        """
        Single-label classifier for piano technique classification
        
        Args:
            embedding_dim (int): Dimension of input embeddings
            embedding_len (int): Length of input embeddings
            model_config (ModelConfig, optional): Model configuration
            training_config (TrainingConfig, optional): Training configuration
        """
        super(SingleLabelClassifier, self).__init__()
        
        # Use default configs if not provided
        self.model_config = model_config or ModelConfig()
        self.training_config = training_config or TrainingConfig()
        
        # Model architecture
        self.model = AudioCNNTransformer(
            h=embedding_dim,
            w=embedding_len,
            n_segs=self.training_config.n_segs,
            out_classes=self.training_config.num_classes,
            config=self.model_config
        )
        
        # Loss function for single-label classification
        self.criterion = nn.CrossEntropyLoss()
        
        # Metrics
        self.accuracy = torchmetrics.Accuracy(
            num_classes=self.training_config.num_classes,
            average='macro',
            task='multiclass'
        )
        self.f1 = torchmetrics.F1Score(
            num_classes=self.training_config.num_classes,
            average='macro',
            task='multiclass'
        )
        self.precision = torchmetrics.Precision(
            num_classes=self.training_config.num_classes,
            average='macro',
            task='multiclass'
        )
        self.recall = torchmetrics.Recall(
            num_classes=self.training_config.num_classes,
            average='macro',
            task='multiclass'
        )
        
    def forward(self, x):
        x = self.model(x)
        return torch.softmax(x, dim=-1)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        
        # Update metrics
        self.accuracy(y_hat, y)
        self.f1(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_accuracy', self.accuracy)
        self.log('val_f1', self.f1)
        self.log('val_precision', self.precision)
        self.log('val_recall', self.recall)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        return optimizer
        
    def save_model(self, path: str):
        torch.save(self.state_dict(), path)
        
    def load_model(self, path: str):
        self.load_state_dict(torch.load(path))
        
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            y_hat = self(x)
            return torch.argmax(y_hat, dim=1) 