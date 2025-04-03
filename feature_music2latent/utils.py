import torch
import torchaudio
import music2latent
import librosa

import torch
import pytorch_lightning as pl
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, TensorDataset


def get_features_and_labels(track_ids, dataset):
    """
    Processes a collection of audio tracks, resamples audio to the required sampling rate,
    and returns their features and labels as PyTorch tensors.

    Parameters
    ----------
    track_ids : list
        List of track IDs (e.g., train_ids, test_ids, val_ids).
    dataset : object
        The dataset object containing the audio tracks.
    music2latent : function
        A function to extract features from the audio data.
    genre_mapping : dict
        A mapping from genre strings to numerical labels.

    Returns
    -------
    feature_matrix : torch.Tensor
        A tensor containing the extracted features for all tracks in the collection.

    label_array : torch.Tensor
        A tensor containing the numerical genre labels for all tracks.
    """
    from music2latent import EncoderDecoder

    encoder = EncoderDecoder()
    labels = []
    features = []
    target_sr = 44100  # Target sampling rate for music2latent

    required_shape = (8192, 320)
    # keep uniform for stacking into tensors matrix, debugged and see shape can be 326, 325

    i = 0

    for track_id in track_ids:
        # Access the track
        track = dataset.track(track_id)
        i += 1
        print(i)

        # Extract audio data (audio array and sampling rate)
        audio, sr = track.audio

        # Resample audio if the sampling rate is not 44.1 kHz
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

        # Compute the feature using music2latent
        feature = encoder.encode(audio, extract_features=True)
        # output 'feature' tensor shape = (channel, dim, sequence length)
        feature = feature[0]
        # dim, sequence length, omit channel info, since audio is mono, channel will be 1

        # cut off a little length to keep shape uniform
        if feature.shape[1] != required_shape[1]:
            print(
                f"Track {track_id} has invalid shape {feature.shape}. Expected {required_shape}. limit the dimension to {required_shape} "
            )
            feature = feature[:, : required_shape[1]]
        print(feature.shape)

        # Get the genre and map to numerical label
        genre = track.genre
        label = genre_mapping.get(genre)  # Default to -1 for unknown genres
        labels.append(label)
        features.append(feature)

    # Convert features and labels to PyTorch tensors
    label_array = torch.tensor(labels, dtype=torch.long)
    feature_matrix = torch.stack(
        [torch.tensor(feature, dtype=torch.float32) for feature in features]
    )

    return feature_matrix, label_array


import os
import torch


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


# 1. Temporal averaging function
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
    return torch.mean(features, dim=2)
