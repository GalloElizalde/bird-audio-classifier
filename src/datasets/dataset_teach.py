import os
import numpy as np
import random
import ast
import torch
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset

class BirdTeachData(Dataset):
    def __init__(self, df):
        self.df_teach_pred = df
        self.dir_soundscapes = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_soundscapes")

    # Convert HH:MM:SS → frame index (samples)
    def time_to_sec(self, t):
        h, m, s = map(int, t.split(':'))
        return (h*3600 + m*60 + s) * 32000

    def __len__(self):
        return len(self.df_teach_pred)
    
    def __getitem__(self, idx):
        # Get rows
        row_pred = self.df_teach_pred.iloc[idx]

        # Create path to load file 
        filename = row_pred["row_id"]
        path_filename = os.path.join(self.dir_soundscapes, filename)

        # Get time to load audio chunk
        start = self.time_to_sec(row_pred["start"])
        end = self.time_to_sec(row_pred["end"])
        
        # Load waveform 
        chunk, sample_rate =  sf.read(path_filename, start=start, stop=end)
        chunk = torch.tensor(chunk, dtype=torch.float32)
        
        # Load target
        target = row_pred.iloc[3:].to_numpy(dtype=np.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return chunk, target
        