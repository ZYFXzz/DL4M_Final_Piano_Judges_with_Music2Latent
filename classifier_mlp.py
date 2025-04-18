import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import List, Dict, Any

class AudioCNNTransformer(nn.Module):
    def __init__(self, h: int, w: int, n_segs: int, out_classes: int, 
                 num_encoder_layers: int = 2, dim_transformer: int = 128,
                 dim_feedforward: int = 64, dropout: float = 0.1, nhead: int = 2):
        """
        Audio CNN-Transformer model
        
        Args:
            h (int): Height of input feature
            w (int): Width of input feature
            n_segs (int): Number of segments
            out_classes (int): Number of output classes
            num_encoder_layers (int): Number of transformer encoder layers
            dim_transformer (int): Dimension of transformer
            dim_feedforward (int): Dimension of feedforward network
            dropout (float): Dropout probability
            nhead (int): Number of attention heads
        """
        super(AudioCNNTransformer, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool = nn.MaxPool2d(2, 2)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_transformer,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layer
        self.fc = nn.Linear(dim_transformer, out_classes)
        
    def forward(self, x):
        # CNN
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Reshape for transformer
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))
        
        # Transformer
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Output
        x = self.fc(x)
        return x

class ClassifierMLP(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any], embedding_dim: int, embedding_len: int):
        """
        Classifier MLP with training and evaluation capabilities
        
        Args:
            cfg (Dict[str, Any]): Configuration dictionary
            embedding_dim (int): Dimension of input embeddings
            embedding_len (int): Length of input embeddings
        """
        super(ClassifierMLP, self).__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.model = AudioCNNTransformer(
            h=embedding_dim,
            w=embedding_len,
            n_segs=cfg['dataset']['n_segs'],
            out_classes=cfg['dataset']['num_classes'],
            **cfg['model']['args']
        )
        
        # Training configuration
        self.learning_rate = cfg['learning_rate']
        self.cfg = cfg
        
        # Set criterion based on task
        if cfg['objective'] == "classification":
            self.criterion = nn.CrossEntropyLoss()
            self.label_dtype = torch.long
        elif cfg['objective'] == "multi-label classification":
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(cfg['dataset']['pos_weight']) * 1.4
            )
            self.label_dtype = torch.long
        else:  # regression
            self.criterion = nn.MSELoss()
            self.label_dtype = torch.float32
            
        # Metrics
        if cfg['task'] == 'technique':
            self.accuracy = torchmetrics.Accuracy(
                num_classes=cfg['dataset']['num_classes'],
                average='macro',
                task='multiclass'
            )
            self.f1 = torchmetrics.F1Score(
                num_classes=cfg['dataset']['num_classes'],
                average='macro',
                task='multiclass'
            )
        else:
            self.precision = torchmetrics.Precision(
                num_classes=cfg['dataset']['num_classes'],
                average='macro',
                task='multiclass'
            )
            self.recall = torchmetrics.Recall(
                num_classes=cfg['dataset']['num_classes'],
                average='macro',
                task='multiclass'
            )
            self.f1 = torchmetrics.F1Score(
                num_classes=cfg['dataset']['num_classes'],
                average='macro',
                task='multiclass'
            )
            self.accuracy = torchmetrics.Accuracy(
                num_classes=cfg['dataset']['num_classes'],
                average='macro',
                task='multiclass'
            )
            
    def forward(self, x):
        x = self.model(x)
        if self.cfg['objective'] == "classification":
            x = torch.softmax(x, dim=-1)
        elif self.cfg['objective'] == "multi-label classification":
            x = torch.sigmoid(x)
        return x
        
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
        if self.cfg['objective'] == "classification":
            self.accuracy(y_hat, y)
            self.f1(y_hat, y)
            self.log('val_accuracy', self.accuracy)
            self.log('val_f1', self.f1)
        elif self.cfg['objective'] == "multi-label classification":
            self.precision(y_hat, y)
            self.recall(y_hat, y)
            self.f1(y_hat, y)
            self.log('val_precision', self.precision)
            self.log('val_recall', self.recall)
            self.log('val_f1', self.f1)
            
        self.log('val_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.get('weight_decay', 0.0)
        )
        return optimizer
        
    def save_model(self, path: str):
        """
        Save the model
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
        
    def load_model(self, path: str):
        """
        Load the model
        
        Args:
            path (str): Path to the model file
        """
        self.load_state_dict(torch.load(path)) 