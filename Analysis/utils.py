import os, sys, math, itertools
import torch
import torchaudio
# from music2latent import EncoderDecoder
import librosa

import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torchmetrics
from torchmetrics import Metric, AveragePrecision, AUROC
import torch.distributed as dist
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    multilabel_confusion_matrix,
)

import matplotlib.pyplot as plt
import itertools
import seaborn as sns

from collections import Counter
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio


import pandas as pd
import numpy as np
import random
from pathlib import Path

LABEL_COLUMNS = [
    "Scales",
    "Arpeggios",
    "Ornaments",
    "Repeatednotes",
    "Doublenotes",
    "Octave",
    "Staccato",
]

N_MELS = 128  # 64 - 128
HOP_LENGTH = 160


def filter_single_label_tracks(dataloader):
    filtered_metadata = []

    for idx in range(len(dataloader)):
        data = dataloader[idx]
        label = data["label"]
        if sum(label) == 1:  # Keep only single-labeled tracks
            filtered_metadata.append(dataloader.metadata.iloc[idx])

    # Create a new TechniqueDataloader instance with the filtered metadata
    filtered_dataloader = TechniqueDataloader(
        mode=dataloader.mode,  # Keep the same mode
        split_ratio=1.0,  # Use all filtered data
        rs=None,  # Random state isn't needed here
        label=dataloader.label,  # Keep the same label type
        technique_dir=dataloader.technique_dir,
    )

    # Replace the metadata in the new dataloader instance
    filtered_dataloader.metadata = pd.DataFrame(filtered_metadata)

    return filtered_dataloader


def plot_label_distribution(dataloader, label_columns, title):
    # Count labels
    from collections import Counter

    label_totals = Counter()
    for idx in range(len(dataloader)):
        data = dataloader[idx]
        label_totals.update(
            {label: value for label, value in zip(label_columns, data["label"])}
        )

    # Prepare counts in order of label_columns
    label_counts = [label_totals.get(label, 0) for label in label_columns]

    # Plot bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(label_columns, label_counts, color="skyblue", edgecolor="black")

    # Add counts above each bar
    for bar, count in zip(bars, label_counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(count),
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    # Add labels and grid
    plt.title(title)
    plt.xlabel("Labels")
    plt.ylabel("Number of Tracks")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def display_track_spec_audio(track, label_columns=LABEL_COLUMNS):
    """
    Generates and displays the mel spectrogram for a given track and plays the audio.

    Parameters:
    track (dict): A dictionary containing the audio path and label.
    label_columns (list): List of label names corresponding to binary labels.

    Returns:
    IPython.display.Audio: The audio playback for the track.
    """
    audio_path = track["audio_path"]
    label = track["label"]

    # Map the binary label to its corresponding names
    label_names = [name for name, value in zip(label_columns, label) if value == 1]
    label_names_str = ", ".join(label_names)

    # Load the audio file
    audio, sr = librosa.load(audio_path, sr=None, mono=True)

    # Compute the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, hop_length=512
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram)

    # Plot the mel spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        mel_spectrogram,
        sr=sr,
        hop_length=512,
        x_axis="time",
        y_axis="mel",
        cmap="viridis",
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Mel Spectrogram - {label_names_str}")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Frequency")
    plt.xlim(0, (len(audio) / sr) / 2)
    plt.tight_layout()
    plt.show()

    # Play the audio file
    return Audio(audio, rate=sr)


class TechniqueClassifier(pl.LightningModule):
    """
    A classifier for piano technique detection (7 classes).
    - Handles multi-label classification for piano techniques
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

        # Fully connected layers for classification (same as GenreClassifier)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),  # Output layer (logits)
        )

        # Multi-label metrics
        self.train_accuracy = torchmetrics.Accuracy(
            task="multilabel", num_labels=num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multilabel", num_labels=num_classes
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="multilabel", num_labels=num_classes
        )

        # Average precision metrics
        self.val_map = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes
        )
        self.val_ap_classes = torchmetrics.AveragePrecision(
            task="multilabel", num_labels=num_classes, average=None
        )
        # Area Under the Receiver Operating Characteristic Curve
        self.val_auc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes)
        self.test_auc = torchmetrics.AUROC(task="multilabel", num_labels=num_classes)

    def forward(self, x):
        return self.mlp(x)  # Return logits

    def training_step(self, batch, batch_idx):
        print("training")
        inputs, labels = self._process_batch(batch)

        # Forward pass
        logits = self(inputs)

        # Loss calculation using BCEWithLogitsLoss for multi-label
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # Calculate accuracy (predictions threshold at 0.5)
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = self.train_accuracy(preds, labels)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        # Unpack batch
        inputs, labels = self._process_batch(batch)

        # Forward pass
        logits = self(inputs)

        # Loss calculation
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        # Get predictions (threshold at 0.5)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Update metrics
        acc = self.val_accuracy(preds, labels)
        self.val_map.update(probs, labels)
        self.val_auc.update(probs, labels)
        self.val_ap_classes.update(probs, labels)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def on_validation_epoch_end(self):
        # Calculate and log mAP
        map_score = self.val_map.compute()
        self.log("val_mAP", map_score, prog_bar=True)

        # Calculate and log per-class AP
        ap_classes = self.val_ap_classes.compute()
        # calculate AUC
        auc_score = self.val_auc.compute()
        self.log("val_auc", auc_score, prog_bar=True)

        # Log per-class metrics
        for i, label in enumerate(self.class_labels):
            if i < len(ap_classes) and not torch.isnan(ap_classes[i]):
                self.log(f"val_AP/{label}", ap_classes[i], prog_bar=False)

    def test_step(self, batch, batch_idx):
        # Unpack batch
        inputs, labels = self._process_batch(batch)

        # Forward pass
        logits = self(inputs)

        # Loss calculation
        # loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        print(logits.dtype, labels.dtype)
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        # Calculate accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = self.test_accuracy(preds, labels)
        # Create test AUC calculation
        probs = torch.sigmoid(logits)
        self.test_auc.update(probs, labels)

        # Log metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def _process_batch(self, batch):
        """Helper method to process batch data from the TechniqueDataloader"""
        inputs, labels = batch

        labels = labels

        return inputs, labels

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
        # Split the metadata into train, validation, and test sets
        # first split into train and test, then split train into train and val
        train_split_index = int(len(metadata) * split_ratio)
        val_split_index = int(train_split_index * 0.75)

        if mode == "train":
            train_pieces = metadata[:val_split_index]
            self.metadata = train_pieces
        elif mode == "val":
            val_pieces = metadata[val_split_index:train_split_index]
            self.metadata = val_pieces
        elif mode == "test":
            test_pieces = metadata[train_split_index:]
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


def get_spectrogram_and_labels(dataloader, augment=False):
    """
    Processes a collection of audio tracks, pads to 5 minutes, segments into 10-second chunks,
    and returns their features and labels as PyTorch tensors.

    If augment=True, applies multiple augmentation strategies to increase dataset size.
    """
    labels = []
    features = []
    # target sampling rate for spectrogram, set to 24000
    target_sr = 24000

    # Compute parameters for segmentation
    segment_duration = 10  # 10 seconds per segment
    total_duration = 60  # 5 minutes (300 seconds) total, 280 to reduce tensor dim, audio mostly around 270s
    num_segments = total_duration // segment_duration  # 30 segments

    # Frame rate is 150 Hz (based on hop_length of 160 samples at 24kHz)
    # 10 seconds at 150 Hz = 1500 frames
    hop_length = HOP_LENGTH
    # frames_per_segment = target_sr // hop_length
    frames_per_segment = int(segment_duration * target_sr / hop_length)

    # Define augmentation strategies
    augmentation_strategies = [
        # 1: Higher pitch, faster
        {"pitch_shift": 2, "time_stretch": 0.9},
        # 2: Lower pitch, slower
        {"pitch_shift": -2, "time_stretch": 1.1},
        # 3: lower pitch,slower
        {
            "pitch_shift": -1,
            "time_stretch": 1.2,
        },
        # 4 : lower pitch, faster
        {
            "pitch_shift": -2,
            "time_stretch": 0.95,
        },
    ]

    for idx in range(len(dataloader)):
        # Load the data and extract audio_path and label
        sample = dataloader[idx]
        audio_path = sample["audio_path"]
        label = sample["label"]
        # Convert multi-label vector to a single integer label
        label = np.argmax(label)
        print(f"Label for track: {label}")
        print(f"Processing track {idx+1}/{len(dataloader)}: {audio_path}")

        try:
            # Load and preprocess audio
            audio, sr = librosa.load(audio_path, sr=None)
            print(f"Original audio shape: {audio.shape}")

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
                sr = target_sr

            # Process the original audio
            processed_audio = process_audio(
                audio,
                sr,
                total_duration,
                segment_duration,
                num_segments,
                frames_per_segment,
            )
            features.append(processed_audio)
            labels.append(label)
           

            # Apply augmentations if requested
            if augment:

                for i, strategy in enumerate(augmentation_strategies):
                    try:
                        print(f"Applying augmentation strategy {i+1}")
                        # Apply pitch shift and time stretch
                        augmented_audio = apply_augmentation(audio, sr, strategy)

                        # Process the augmented audio
                        processed_aug_audio = process_audio(
                            augmented_audio,
                            sr,
                            total_duration,
                            segment_duration,
                            num_segments,
                            frames_per_segment,
                        )

                        # Add to our dataset
                        features.append(processed_aug_audio)
                        labels.append(label)
                    except Exception as e:
                        print(f"Error during augmentation {i+1}: {e}")
                        continue
            # break
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

    # Convert to PyTorch tensors and shuffle (to avoid similar examples in sequence)
    features_tensor = torch.tensor(features)
    labels_tensor = torch.tensor(labels)

    # Optional: shuffle the data
    if augment:
        print("Shuffling augmented dataset...")
        shuffle_indices = torch.randperm(len(features_tensor))
        features_tensor = features_tensor[shuffle_indices]
        labels_tensor = labels_tensor[shuffle_indices]

    print("Execution completed, saving features and tensor as .pt file")
    print(f"Feature tensor shape: {features_tensor.shape}")
    print(f"Label tensor shape: {labels_tensor.shape}")

    # Save features and labels
    save_features_and_labels(
        features=features_tensor,
        labels=labels_tensor,
        split_name=dataloader.mode,
        folder_name="spectrogram",
        augment=augment,
    )

    return features_tensor, labels_tensor


def apply_augmentation(audio, sr, strategy):
    """Apply audio augmentation according to the specified strategy"""
    # Make a copy to avoid modifying the original
    augmented = audio.copy()

    # Apply time stretching first
    if "time_stretch" in strategy and strategy["time_stretch"] != 1.0:
        rate = strategy["time_stretch"]
        augmented = librosa.effects.time_stretch(augmented, rate=rate)

    # Then apply pitch shifting (in semitones)
    if "pitch_shift" in strategy and strategy["pitch_shift"] != 0:
        n_steps = strategy["pitch_shift"]
        augmented = librosa.effects.pitch_shift(augmented, sr=sr, n_steps=n_steps)

    return augmented


def process_audio(
    audio, sr, total_duration, segment_duration, num_segments, frames_per_segment
):
    """Process audio: pad/truncate, segment, and compute spectrograms"""
    # First pad the audio to exactly 5 minutes (300 seconds)
    target_length = total_duration * sr  # 5 minutes (300s) of audio
    print(f"Target length in samples: {target_length}")
    if len(audio) < target_length:
        # If shorter than required length, pad with zeros
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode="constant")
    else:
        # If longer than required length, truncate
        audio = audio[:target_length]

    # Segment the padded audio into 30 segments of 10 seconds each
    track_features = []
    segment_samples = segment_duration * sr

    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = audio[start:end]

        # Compute mel spectrogram for this segment
        mel_db = compute_mel_spectrogram(
            segment, sr, frames_per_segment=frames_per_segment
        )
        track_features.append(mel_db)

    # Stack segments into a single array of shape (30, 128, 1500)
    track_features = np.stack(track_features)

    # Reshape to match (30, F×10, D) format: (30, 1500, 128)
    # Need to transpose from (30, 128, 1500) to (30, 1500, 128)
    track_features = np.transpose(track_features, (0, 2, 1))

    return track_features


def compute_mel_spectrogram(
    y, sr, n_fft=400, hop_length=HOP_LENGTH, n_mels=N_MELS, frames_per_segment=1500
):
    """
    Compute mel spectrogram for a segment, ensuring consistent output dimensions.
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window="hann",
        fmax=sr / 2,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Ensure exactly frames_per_segment frames
    if mel_db.shape[1] > frames_per_segment:
        mel_db = mel_db[:, :frames_per_segment]
    elif mel_db.shape[1] < frames_per_segment:
        padding = frames_per_segment - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, padding)), mode="constant")

    return mel_db


def save_features_and_labels(features, labels, split_name, folder_name, augment=False):
    """
    Saves the features and labels as PyTorch files in a structured directory format.

    This function saves the provided `features` and `labels` tensors as `.pt` files within a designated folder. It ensures that the directory structure exists, creating it if necessary. The saved files are named using the provided `split_name` parameter to allow differentiation between datasets (e.g., 'train', 'val', 'test').

    Parameters
    ----------
    features : torch.Tensor
        Tensor containing feature data to be saved.
    labels : torch.Tensor
        Tensor containing label data corresponding to the features.
    split_name : str
        Identifier for the data split (e.g., 'train', 'val', 'test').
    folder_name : str, optional
        Parent folder where the files will be saved (default: "latent_data").

    Returns
    -------
    None
        This function does not return any values but saves the tensors as files in the specified directory.

    Raises
    ------
    FileNotFoundError
        If the `folder_name` directory cannot be created due to file system constraints.
    ValueError
        If `split_name` is empty or contains invalid characters.

    Example
    -------
    >>> save_features_and_labels(features_tensor, labels_tensor, "train")
    This will save the tensors `features_tensor` and `labels_tensor` as "train_features.pt" and "train_labels.pt" within the default "latent_data" directory.
    """

    # Ensure parent directory paths are structured correctly

    if augment:
        folder_name = folder_name + "_augmented"

    project_dir = Path.cwd().parents[0]
    print(f"Project directory: {project_dir}")
    save_dir = project_dir / "computed_features" / folder_name
    print(f"Saving features and labels to: {save_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save features and labels tensors
    torch.save(features, save_dir / f"{split_name}_features.pt")
    torch.save(labels, save_dir / f"{split_name}_labels.pt")
    print(
        f"Features and labels saved successfully for split '{split_name}' in {save_dir}."
    )


def load_features_and_labels(split_name, folder_name="spectrogram", augment=False):
    """
    Loads the features and labels from PyTorch files in a structured directory format.

    This function retrieves the `features` and `labels` tensors saved in `.pt` format for a specific split (e.g., 'train', 'val', 'test'). It assumes the directory structure and file naming conventions are consistent with the `save_features_and_labels` function.

    Parameters
    ----------
    split_name : str
        Name of the split (e.g., 'train', 'val', 'test').
    folder_name : str, optional
        Parent folder where the files are stored (default: "spectrogram").

    Returns
    -------
    features : torch.Tensor
        The loaded features tensor.
    labels : torch.Tensor
        The loaded labels tensor.

    Raises
    ------
    FileNotFoundError
        If the specified feature or label file does not exist.
    """
    from pathlib import Path
    import torch

    if augment:
        folder_name = folder_name + "_augmented"

    # Ensure parent directory paths are structured correctly
    project_dir = Path.cwd().parents[0]
    save_dir = project_dir / "computed_features" / folder_name

    # Load features and labels tensors
    features_path = save_dir / f"{split_name}_features.pt"
    labels_path = save_dir / f"{split_name}_labels.pt"
    print(f"Loading features and labels from: {save_dir}")

    # Check if files exist before loading
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    features = torch.load(features_path)
    labels = torch.load(labels_path)

    print(
        f"Features and labels successfully loaded for split '{split_name}' from {save_dir}."
    )

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
