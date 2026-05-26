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

class BirdSoundscapeDataset(Dataset):
    '''
    Inputs:
    - df_soundscapes ("train_soundscapes_labels.csv" dataframe with filename, start, end, target information)

    Outputs:
    - chunk: Tensor [samples]
    - target: Tensor [num_classes]

    Description:
    Loads audio chunks from soundscapes using timestamps and returns
    waveform with multi-hot labels.
    '''
    def __init__(self, df_soundscapes):
        self.df_soundscapes = df_soundscapes
        self.dir_soundscapes = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_soundscapes")
        self.utils = UtilFunctions()

    # Convert HH:MM:SS → frame index (samples)
    def time_to_sec(self, t):
        h, m, s = map(int, t.split(':'))
        return (h*3600 + m*60 + s) * 32000

    def __len__(self):
        return len(self.df_soundscapes)
    
    def __getitem__(self, idx):
        # Get rows
        row = self.df_soundscapes.iloc[idx]

        # Create path to load file 
        filename = row["filename"]
        path_filename = os.path.join(self.dir_soundscapes, filename)

        # Get time to load audio chunk
        start = self.time_to_sec(row["start"])
        end = self.time_to_sec(row["end"])

        # Load waveform 
        chunk, sample_rate =  sf.read(path_filename, start=start, stop=end)
        chunk = torch.tensor(chunk, dtype=torch.float32)
        
        # Get target from string
        target = self.utils.multi_hot_soundscapes(row["primary_label"])
        target = torch.tensor(target, dtype=torch.float32)

        # Data
        data = [filename, row["start"], row["end"]]

        return chunk, target, data
    

# SANITY CHECK
if __name__ == "__main__":
    import random

    # Load dataframe
    df = pd.read_csv(os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_soundscapes_labels.csv"))

    print("Loading Dataset...")
    dataset = BirdSoundscapeDataset(df)

    # Sample a few random items
    for _ in range(5):
        idx = random.randint(0, len(dataset) - 1)
        waveform, target, data = dataset[idx]

        print(f"\nIndex: {idx}")
        print(f"Waveform shape: {waveform.shape}, dtype: {waveform.dtype}")
        print(f"Target shape: {target.shape}, sum: {target.sum()}")
        print(f"Data {data}")

        # Basic sanity checks
        assert target.ndim == 1
        assert torch.all((target == 0) | (target == 1))
        assert target.sum() >= 1

        # Check consistency with number of classes
        assert target.shape[0] == dataset.utils.n_classes

        print(f"Active labels idx: {target.nonzero().squeeze()}")

    print("\nSanity check passed.")