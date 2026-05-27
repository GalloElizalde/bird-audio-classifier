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
        - df_soundscapes: dataframe from "train_soundscapes_labels.csv" with filename, start, end and labels

    Outputs:
        - chunk: Tensor [audio samples]
        - target: Tensor [num_classes]
        - data: list with filename, start and end

    Description:
        Loads audio chunks from soundscapes using timestamps and returns
        waveform, multi-hot target and chunk metadata.
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
    
