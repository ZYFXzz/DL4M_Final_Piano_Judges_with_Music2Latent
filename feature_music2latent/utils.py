import os, sys, math, itertools
import torch
import torchaudio
from music2latent import EncoderDecoder
import librosa

import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset, Dataset


import pandas as pd
import numpy as np
import random


import pandas as pd


## technique data loader objects is copied from the original piano judge repo, with some modifications
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class TechniqueClassifier(pl.LightningModule):
    """
    A multi-label classifier for piano technique detection (7 classes).
    Based on the PianoJudge paper approach for technique identification.
    - Tracks metrics for multi-label classification including mAP and AUC
    - Supports both multi-label and single-label prediction modes
    - Uses Adam optimizer with ReduceLROnPlateau scheduler
    """

    def __init__(self, input_dim=768, num_classes=7, learning_rate=1e-3, threshold=0.5):
        super().__init__()
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.save_hyperparameters()

        # Fully connected layers for classification
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),  # Output layer (logits)
        )

        # Initialize metrics
        self.train_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes)
        self.val_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes)
        self.test_auroc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes)

        self.train_map = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes
        )
        self.val_map = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes
        )
        self.test_map = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes
        )

    def forward(self, x):
        return self.mlp(x)  # Return logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)  # Forward pass
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        # Compute accuracy (all binary predictions match)
        acc = (preds == labels).all(dim=1).float().mean()

        # Update AUC and mAP metrics
        self.train_auroc(probs, labels.int())
        self.train_map(probs, labels.int())

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        # Compute accuracy (all binary predictions match)
        acc = (preds == labels).all(dim=1).float().mean()

        # Update AUC and mAP metrics
        self.val_auroc(probs, labels.int())
        self.val_map(probs, labels.int())

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        # Compute accuracy (all binary predictions match)
        acc = (preds == labels).all(dim=1).float().mean()

        # Update AUC and mAP metrics
        self.test_auroc(probs, labels.int())
        self.test_map(probs, labels.int())

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def on_train_epoch_end(self):
        # Log metrics at the end of each epoch
        self.log("train_auroc", self.train_auroc.compute(), prog_bar=True)
        self.log("train_map", self.train_map.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        # Log metrics at the end of each epoch
        self.log("val_auroc", self.val_auroc.compute(), prog_bar=True)
        self.log("val_map", self.val_map.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        # Log metrics at the end of each epoch
        self.log("test_auroc", self.test_auroc.compute(), prog_bar=True)
        self.log("test_map", self.test_map.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=3,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def predict_step(self, batch, batch_idx):
        # Handle both cases: batch as a tuple/list or as a direct tensor
        if isinstance(batch, (tuple, list)):
            inputs = batch[0]
        else:
            inputs = batch

        # Make sure inputs is a tensor (not a list)
        if not isinstance(inputs, torch.Tensor):
            raise TypeError(f"Expected inputs to be a tensor, got {type(inputs)}")

        # Move inputs to the same device as the model parameters
        device = next(self.parameters()).device
        inputs = inputs.to(device)

        # Forward pass
        logits = self(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > self.threshold).float()

        return {"logits": logits, "probabilities": probs, "predictions": preds}


# Helper function to process input data
def process_audio_embeddings(embedding, max_length=30):
    """
    Process audio embeddings to ensure consistent shape
    Args:
        embedding: Audio embedding from encoder
        max_length: Maximum number of segments to include (default 30 = 5 minutes)
    """
    # Pad or truncate embedding to desired length
    if embedding.shape[0] > max_length:
        return embedding[:max_length]
    elif embedding.shape[0] < max_length:
        padding = torch.zeros(
            (max_length - embedding.shape[0], embedding.shape[1], embedding.shape[2])
        )
        return torch.cat([embedding, padding], dim=0)
    else:
        return embedding


class TechniqueDataloader:
    def __init__(
        self, mode="train", split_ratio=0.8, rs=42, label="multi", technique_dir=None
    ):
        if technique_dir is None:
            raise ValueError("Please provide a valid path for 'technique_dir'")
        # Read the metadata CSV file
        technique_csv = os.path.join(technique_dir, "metadata.csv")  ## os load csv
        metadata = pd.read_csv(technique_csv)
        metadata = metadata.sample(frac=1, random_state=rs)

        self.mode = mode
        self.label = label
        self.technique_dir = technique_dir
        self.label_columns = [
            "Scales",
            "Arpeggios",
            "Ornaments",
            "Repeatednotes",
            "Doublenotes",
            "Octave",
            "Staccato",
        ]

        split_index = int(len(metadata) * split_ratio)

        if mode == "train":
            train_pieces = metadata[:split_index]
            self.metadata = train_pieces
        elif mode == "test":
            test_pieces = metadata[split_index:]
            self.metadata = test_pieces

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):

        p = self.technique_dir + "/" + self.metadata.iloc[idx]["id"] + ".wav"

        if self.label == "multi":
            labels = list(self.metadata.iloc[idx][self.label_columns])
        elif self.label == "single":
            # labels = np.argmax(self.metadata.iloc[idx][self.label_columns])
            labels = list(self.metadata.iloc[idx][self.label_columns])
        return {"audio_path": p, "label": labels}


def get_features_and_labels(dataloader):
    """
    Processes a collection of audio tracks, resamples audio to the required sampling rate,
    and returns their features and labels as PyTorch tensors.

    Parameters
    ----------
    dataloader : dataloader object defined above


    music2latent : function
        A function to extract features from the audio data.
    genre_mapping : dict
        A mapping from genre strings to numerical labels.

    Returns
    -------
    feature_matrix : torch.Tensor
        A tensor containing the extracted features for all tracks in the collection.

    label_matrix : torch.Tensor
        A tensor containing the numerical genre labels for all tracks.
    """
    # initialize music2latent encoder
    from music2latent import EncoderDecoder

    encoder = EncoderDecoder()

    labels = []
    features = []
    target_sr = 44100  # Target sampling rate for music2latent

    # required_shape = (8192, 320)
    # keep uniform for stacking into tensors matrix, debugged and see shape can be 326, 325

    i = 0

    # for idx in range(len(dataloader)):
    for idx in range(len(dataloader)):

        # load the data and extract audio_path and label
        sample = dataloader[idx]
        audio_path = sample["audio_path"]
        label = sample["label"]
        print(audio_path)
        print(label)
        print(f"Processing track {idx+1}/{len(dataloader)}: {audio_path}")
        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=None)
            print(audio.shape)

            # Convert stereo to mono if needed
            if audio.ndim > 1:
                print(f"Track {idx} is stereo - converting to mono")
                audio = librosa.to_mono(audio)

            # Normalize the audio
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                audio = audio / max_amplitude

            # Resample if necessary
            if sr != target_sr:
                print(f"Track {idx}, resample from {sr} to {target_sr}")
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Compute the feature using music2latent
            # Note: This extracts features from the encoder
            feature = encoder.encode(audio, extract_features=True)

            # Output is (channel, dim, seq_length), we take out channel
            feature = feature[0]  # dim, sequence length
            print(feature.shape)
            feature = temporal_average(feature)
            print(feature.shape)

            print(f"Feature shape: {feature.shape}")

            # Store data
            features.append(feature)
            labels.append(label)

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

        # # if feature.shape[1] != required_shape[1]:
        # #     print(
        # #         f"Track {idx} has invalid shape {feature.shape}. Expected {required_shape}. limit the dimension to {required_shape} "
        # #     )
        # #     feature = feature[:, : required_shape[1]]
        # print(feature.shape)

    # Convert features and labels to PyTorch tensors
    label_matrix = torch.tensor(labels)
    # feature_matrix = torch.stack(
    #     [torch.tensor(feature, dtype=torch.float32) for feature in features]
    # )
    feature_matrix = torch.stack(
        [
            feature.clone().detach().type(torch.float32) for feature in features
        ]  # Use clone().detach() for features， suggested by terminal print
    )
    print("execution completed, saving features and tensor as .pt file")
    # print(dataloader.mode)
    save_features_and_labels(
        features=feature_matrix, labels=label_matrix, split_name=dataloader.mode
    )

    return feature_matrix, label_matrix

    # borrowed template from previous DL homework, saved here just for referencing when needed

    # for track_id in track_ids:
    #     # Access the track
    #     track = dataset.track(track_id)
    #     i += 1
    #     print(i)

    #     # Extract audio data (audio matrix and sampling rate)
    #     audio, sr = track.audio

    #     # Resample audio if the sampling rate is not 44.1 kHz
    #     if sr != target_sr:
    #         audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    #     # Compute the feature using music2latent
    #     feature = encoder.encode(audio, extract_features=True)
    #     # output 'feature' tensor shape = (channel, dim, sequence length)
    #     feature = feature[0]
    #     # dim, sequence length, omit channel info, since audio is mono, channel will be 1

    #     # cut off a little length to keep shape uniform
    #     if feature.shape[1] != required_shape[1]:
    #         print(
    #             f"Track {track_id} has invalid shape {feature.shape}. Expected {required_shape}. limit the dimension to {required_shape} "
    #         )
    #         feature = feature[:, : required_shape[1]]
    #     print(feature.shape)

    #     # Get the genre and map to numerical label
    #     genre = track.genre
    #     label =
    #     labels.append(label)
    #     features.append(feature)

    # # Convert features and labels to PyTorch tensors
    # label_matrix = torch.tensor(labels, dtype=torch.long)
    # feature_matrix = torch.stack(
    #     [torch.tensor(feature, dtype=torch.float32) for feature in features]
    # )

    # return feature_matrix, label_matrix


def save_features_and_labels(features, labels, split_name, folder_name="latent_data"):
    """
    Saves the features and labels as PyTorch files under the specified folder.

    Parameters
    ----------
    features : torch.Tensor
        The features tensor to save.
    labels : torch.Tensor
        The labels tensor to save.
    split_name : str
        Name of the split (e.g., 'train', 'val', 'test').
    folder_name : str, optional
        Name of the folder to save files in (default: "latent_data").
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder_name, exist_ok=True)

    # Save features and labels with the split name
    torch.save(features, os.path.join(folder_name, f"{split_name}_features.pt"))
    torch.save(labels, os.path.join(folder_name, f"{split_name}_labels.pt"))


def load_features_and_labels(split_name, folder_name="latent_data"):
    """
    Loads the features and labels from PyTorch files in the specified folder.

    Parameters
    ----------
    split_name : str
        Name of the split (e.g., 'train', 'val', 'test').
    folder_name : str, optional
        Name of the folder to load files from (default: "latent_data").

    Returns
    -------
    features : torch.Tensor
        The loaded features tensor.
    labels : torch.Tensor
        The loaded labels tensor.
    """
    # Load features and labels with the split name
    features = torch.load(os.path.join(folder_name, f"{split_name}_features.pt"))
    labels = torch.load(os.path.join(folder_name, f"{split_name}_labels.pt"))

    return features, labels


# utilities Temporal averaging function
def temporal_average(features):
    """
    Perform averaging along the feature dimension (D).

    Parameters
    ----------
    features : torch.Tensor
        Tensor with shape (D, T), where D is the feature dimension and T is the time sequence.

    Returns
    -------
    torch.Tensor
    """
    # print(features)
    return torch.mean(features, dim=1)


## classifier
class GenreClassifier(pl.LightningModule):
    """
    A convolutional neural network for genre classification (10 classes).
    Suitable for probing tasks, works with input dimensions of (8192, 320).
    - Tracks loss and accuracy for train/val/test.
    - Uses Adam optimizer with ReduceLROnPlateau scheduler.
    """

    def __init__(self, input_dim=8192, num_classes=10, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate

        # Fully connected layers for classification
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),  # Output layer (logits)
        )

    def forward(self, x):
        x = self.mlp(x)  # Apply fully connected layers
        return x

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)  # Forward pass
        loss = F.cross_entropy(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        logits = self(inputs)
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=3,
            verbose=True,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


# 3. Prepare DataLoader
def create_dataloader(features, labels, batch_size=32):
    """
    Create a PyTorch DataLoader from features and labels.
    Parameters
    ----------
    features : torch.Tensor
        The input features tensor.
    labels : torch.Tensor
        The labels tensor.
    batch_size : int, optional
        Batch size for the DataLoader (default: 32).
    Returns
    -------
    DataLoader
        A PyTorch DataLoader object.
    """
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
