import os, sys, math, itertools
import torch
import torchaudio
from music2latent import EncoderDecoder
import librosa

import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchmetrics
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import classification_report, confusion_matrix


class TechniqueClassifier(pl.LightningModule):
    """
    A classifier for piano technique detection (7 classes).
    - Handles single-label classification for piano techniques
    - Tracks loss and accuracy metrics
    - Uses Adam optimizer with ReduceLROnPlateau scheduler
    """

    def __init__(self, input_dim=8192, num_classes=7, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Class labels for logging purposes
        self.class_labels = [
            "Scales",
            "Arpeggios",
            "Ornaments",
            "Repeatednotes",
            "Doublenotes",
            "Octave",
            "Staccato",
        ]

        # Fully connected layers for classification
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),  # Output layer (logits)
        )

        # Single-label accuracy metric
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.mlp(x)  # Return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = self._process_batch(batch)

        # Forward pass
        logits = self(inputs)

        # Loss calculation using CrossEntropyLoss for single-label classification
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = self._process_batch(batch)

        # Forward pass
        logits = self(inputs)

        # Loss calculation
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = self._process_batch(batch)

        # Forward pass
        logits = self(inputs)

        # Loss calculation
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, labels)

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def _process_batch(self, batch):
        inputs, labels = batch
        return inputs, labels

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
