import librosa
import numpy as np
import torch

def compute_mel_spectrogram(
    audio_path,
    sr=22050,
    n_fft=400,
    hop_length=160,
    n_mels=128,
    max_frames=512
):
    """
    Loads an audio file and returns a fixed-size mel-spectrogram as a torch tensor.
    """
    y, _ = librosa.load(audio_path, sr=sr)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad or crop to fixed length
    if mel_db.shape[1] < max_frames:
        pad_width = max_frames - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_frames]

    return torch.tensor(mel_db).float()  # shape: [128, max_frames]