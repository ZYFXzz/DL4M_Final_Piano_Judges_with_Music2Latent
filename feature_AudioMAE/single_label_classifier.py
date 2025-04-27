import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Any, Optional
from dataclasses import dataclass
from einops import repeat


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dropout,
        padding,
        stride,
        skip=False,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.skip = skip

    def forward(self, x):
        identity = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)

        if self.skip:
            # Skip Connection - Ensuring the channel dimensions match
            if identity.shape != out.shape:
                if identity.shape[1] < out.shape[1]:
                    identity = repeat(
                        identity,
                        f"b c w h -> b (c repeat) w h",
                        repeat=int(out.shape[1] / identity.shape[1]),
                    )
                else:
                    identity = identity[:, : out.shape[1], :, :]

            out += identity
        return out


@dataclass
class ModelConfig:
    """Configuration for the model architecture"""

    num_encoder_layers: int = 2
    dim_transformer: int = 128
    dim_feedforward: int = 64
    dropout: float = 0.1
    nhead: int = 2
    cnn_channels: int = 32
    kernel_size: int = 3
    padding: int = 1
    stride: int = 1


@dataclass
class TrainingConfig:
    """Configuration for training"""

    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_classes: int = 7
    n_segs: int = 10


class AudioCNNTransformer(nn.Module):
    def __init__(
        self, h: int, w: int, n_segs: int, out_classes: int, config: ModelConfig
    ):
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

        # CNN blocks with skip connections
        self.conv_block1 = ConvBlock(
            1,
            config.cnn_channels,
            config.kernel_size,
            config.dropout,
            config.padding,
            config.stride,
        )
        self.conv_block2 = ConvBlock(
            config.cnn_channels,
            1,
            config.kernel_size,
            config.dropout,
            config.padding,
            config.stride,
        )

        # Calculate output dimensions after CNN
        conv_out_h = self.conv_output_shape(
            h, config.kernel_size, config.stride, config.padding, 2
        )
        conv_out_w = self.conv_output_shape(
            w, config.kernel_size, config.stride, config.padding, 2
        )

        # Projection layer to match transformer dimension
        self.fc = nn.Linear(conv_out_h * conv_out_w, config.dim_transformer)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_transformer,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        # Output layer
        self.regression = nn.Linear(config.dim_transformer * n_segs, out_classes)

    def forward(self, x):
        # x shape: (batch, n_segs, seq_len, emb_dim)
        batch_size, n_segs, seq_len, emb_dim = x.shape

        # Apply CNN blocks
        x = x.view(batch_size * n_segs, 1, seq_len, emb_dim)  # Reshape for CNN
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        # Flatten and transpose for Transformer
        x = x.view(batch_size, n_segs, -1).transpose(
            0, 1
        )  # (n_segs, batch, flattened emb_dim)
        x = self.fc(x)

        # Apply Transformer Encoder
        transformer_output = self.transformer_encoder(x)

        # Aggregate the output and reshape
        agg_output = transformer_output.transpose(0, 1).reshape(batch_size, -1)

        # Regression to get final output
        output = self.regression(agg_output).squeeze(-1)
        return output

    def conv_output_shape(self, input_dim, kernel_size, stride, padding, n=1):
        for _ in range(n):
            input_dim = ((input_dim - kernel_size + 2 * padding) / stride) + 1
        return int(input_dim)


class SingleLabelClassifier(pl.LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        embedding_len: int,
        model_config: Optional[ModelConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ):
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
            config=self.model_config,
        )

        # Loss function for single-label classification
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.accuracy = torchmetrics.Accuracy(
            num_classes=self.training_config.num_classes,
            average="macro",
            task="multiclass",
        )
        self.f1 = torchmetrics.F1Score(
            num_classes=self.training_config.num_classes,
            average="macro",
            task="multiclass",
        )
        self.precision = torchmetrics.Precision(
            num_classes=self.training_config.num_classes,
            average="macro",
            task="multiclass",
        )
        self.recall = torchmetrics.Recall(
            num_classes=self.training_config.num_classes,
            average="macro",
            task="multiclass",
        )

    def forward(self, x):
        x = self.model(x)
        return torch.softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.float()  # Ensure y_hat is float for CrossEntropyLoss
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

        # Free unused GPU memory
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.float()  # Ensure y_hat is float for CrossEntropyLoss
        loss = self.criterion(y_hat, y)

        # Update metrics
        self.accuracy(y_hat, y)
        self.f1(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)

        # Log metrics
        self.log("val_loss", loss,prog_bar=True)
        self.log("val_accuracy", self.accuracy,prog_bar=True)
        self.log("val_f1", self.f1)
        self.log("val_precision", self.precision)
        self.log("val_recall", self.recall)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.float()  # Ensure y_hat is float for CrossEntropyLoss
        loss = self.criterion(y_hat, y)

        # Update metrics
        self.accuracy(y_hat, y)
        self.f1(y_hat, y)
        self.precision(y_hat, y)
        self.recall(y_hat, y)

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_accuracy", self.accuracy)
        self.log("test_f1", self.f1)
        self.log("test_precision", self.precision)
        self.log("test_recall", self.recall)

        return {"test_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
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
