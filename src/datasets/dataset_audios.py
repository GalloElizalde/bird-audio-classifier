import os
import numpy as np
import random
import ast
import torch
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torchaudio.functional as AF
from utils.util_functions import UtilFunctions


class BirdAudioDataset(Dataset):
    '''
    Inputs:
        - df: transformed dataframe of "train.csv" containing name of file, target and k-Folds index
        - segment_seconds: final duration of audio in seconds after cutting it
        - p_noise: prob. of adding noise
        - p_filter: prob. of adding random equalization to boost or attenuate different frequency bands in the audio.
        - p_mix: prob. of adding a secondary audio
        - p_sounscapes_noies: probability of adding a soundscape audio as noise over a ESC-50 audio
        - alpha: factor of noise [0,1]

    Outputs:
        - chunk: Tensor [audio sample]
        - target: Tensor [num_classes]

    Description:
    Loads audio, extracts fixed-length segment, applies augmentation,
    returns waveform and multi-label target.
    '''

    def __init__(self, df, segment_seconds = 5, p_noise = 0.5, p_filter = 0.25, p_mix = 0.66, p_soundscape_noise = 0.5, alpha = 0.3):
        self.df = df.copy()
        self.num_samples = len(self.df)
        self.segment_seconds = segment_seconds  # to cut audio in segments of 5s

        self.utils = UtilFunctions()
        
        self.path_audios = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_audio")
        self.audio_files = os.listdir(self.path_audios)
        
        self.path_soundscapes = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_soundscapes")
        self.soundscape_files = os.listdir(self.path_soundscapes)
        
        self.path_esc50 = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_esc50")
        self.esc50_files = os.listdir(self.path_esc50)

        self.p_noise = p_noise
        self.p_filter = p_filter
        self.p_mix = p_mix
        self.p_soundscape_noise = p_soundscape_noise
        self.alpha = alpha

    # Function to cut audio into "segment_seconds"
    def cut_audio(self, waveform, sample_rate=32000):
        # Verify if audio is larger than segment_seconds (5s)
        length_waveform = len(waveform)
        segment_samples = int(self.segment_seconds * sample_rate)
        
        if length_waveform <= segment_samples:
            # Add missing seconds as 0-values(mode=constant) o reflections (mode=reflect)
            missing_segment = segment_samples - length_waveform
            # Pad equally on both sides
            pad_left = missing_segment // 2
            pad_right = missing_segment - pad_left
            waveform = np.pad(waveform, (pad_left, pad_right), mode="constant")  
        else:
            # Cut a 5s chunk randomly from waveform
            start = np.random.randint(0, (length_waveform - segment_samples)+1)
            end = start + segment_samples
            waveform = waveform[start:end]
        return waveform    

    # Function to add noise from train soundscapes or ESC50
    def random_add_noise(self, waveform):
        # Random trial to add noise
        if random.random() < self.p_noise:
            # Random trial to select source of noise
            if random.random() < self.p_soundscape_noise:
                # Select noise from soundscapes 
                path = self.path_soundscapes
                random_filename = random.choice(self.soundscape_files)
                path_filename = os.path.join(path, random_filename)
                noise, sample_rate = sf.read(path_filename)
                if noise.ndim == 2:
                    noise = noise.mean(axis=1)
                noise_chunk = self.cut_audio(noise, sample_rate)
            else:
                # Select noise from ESC50
                path = self.path_esc50
                random_filename = random.choice(self.esc50_files)
                path_filename = os.path.join(path, random_filename)
                noise, sample_rate = sf.read(path_filename)
                if noise.ndim == 2:
                    noise = noise.mean(axis=1)
                if sample_rate != 32000:
                    noise = torch.tensor(noise, dtype=torch.float32)
                    noise = AF.resample(noise, sample_rate, 32000).numpy()
                    sample_rate = 32000
                noise_chunk = self.cut_audio(noise, sample_rate)
            # add noise to waveform controling signal to noise ratio
            rms_waveform = np.sqrt(np.mean(waveform**2)) + 1e-8
            rms_noise = np.sqrt(np.mean(noise_chunk**2)) + 1e-8
            noise_chunk = noise_chunk * (rms_waveform / rms_noise)
            waveform = waveform + (self.alpha * noise_chunk)
        return waveform

    # Function to Random filtering
    def random_filter(self, waveform, sample_rate=32000):
        if random.random() > self.p_filter:
            return waveform
        waveform = torch.tensor(waveform, dtype=torch.float32)
        waveform = AF.equalizer_biquad(waveform.unsqueeze(0), sample_rate=sample_rate, 
                        center_freq=random.uniform(500, 8000), gain=random.uniform(-6, 6), Q=random.uniform(0.5, 1.5)).squeeze(0)
        return waveform.numpy()
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Get rows
        row_1 = self.df.iloc[idx]
        
        # Load audio filename and create path to file 
        filename = row_1["filename"]
        path_filename = os.path.join(self.path_audios, filename)

        # Extract waveform and 5s chunk
        waveform_1, _ = sf.read(path_filename)
        if waveform_1.ndim == 2:
            waveform_1 = waveform_1.mean(axis=1)
        chunk_1 = self.cut_audio(waveform_1)

        # Get target
        target_1 = torch.tensor(self.utils.multi_hot_audios(row_1["primary_label"], row_1["secondary_labels"]), dtype=torch.float32)
 
        # Return chunk and target or randomly apply MixUP(add a second waveform to first one randomly)
        if random.random() > self.p_mix:
            # Add background mixing, Filtering and convert to torch
            chunk_1 = self.random_add_noise(chunk_1)
            chunk_1 = self.random_filter(chunk_1)
            chunk_1 = torch.tensor(chunk_1, dtype = torch.float32)
            return chunk_1, target_1
        
        # MixUp
        idx_2 = random.randint(0, self.num_samples - 1)  
        row_2 = self.df.iloc[idx_2]
        waveform_2, _ = sf.read(os.path.join(self.path_audios, row_2["filename"]))
        if waveform_2.ndim == 2:
            waveform_2 = waveform_2.mean(axis=1)
        
        # Get chunk and target of second waveform
        chunk_2 = self.cut_audio(waveform_2)
        target_2 = torch.tensor(self.utils.multi_hot_audios(row_2["primary_label"], row_2["secondary_labels"]),dtype=torch.float32)

        # Combine target and chunks
        target = torch.maximum(target_1, target_2)
        chunk = chunk_1 + chunk_2

        # Add noise  chunk and transform to tensor:
        chunk = self.random_add_noise(chunk)
        chunk = self.random_filter(chunk)
        chunk = torch.tensor(chunk, dtype = torch.float32)

        return chunk, target
    


#   ============================ SANITY CHECK  =============================================
if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "data", "df_with_folds.csv")

    df_train = pd.read_csv(DATA_PATH)

    # Create dataset
    print("Loading Data")
    dataset_train = BirdAudioDataset(df_train, segment_seconds = 5, p_noise = 1, p_filter = 0.25, 
                                     p_mix = 0.66, p_soundscape_noise = 0, alpha = 0.3)

    # Check a few random items
    for _ in range(10):
        idx = random.randint(0, len(dataset_train) - 1)
        x, y = dataset_train[idx]

        print(f"idx={idx}, x shape={x.shape}, y sum={y.sum()}")

        # Basic sanity checks
        assert x.shape[0] == 5 * 32000
        assert y.ndim == 1
        assert torch.all((y == 0) | (y == 1))
        assert y.sum() >= 1

    print("Sanity check passed.")
#============================================================================================
