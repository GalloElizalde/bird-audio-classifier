import os
import numpy as np
import random
import ast
import torch
import soundfile as sf
import pandas as pd
from torch.utils.data import Dataset

class BirdTeachData(Dataset):
    def __init__(self, df_real, df_teacher):
        # Load real targets and teacher predictions
        self.df_real = df_real.reset_index(drop=True)
        self.df_teacher = df_teacher.reset_index(drop=True)

        # Directory with soundscapes
        self.dir_soundscapes = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_soundscapes")

    # Convert HH:MM:SS → frame index
    def time_to_sec(self, t):
        h, m, s = map(int, str(t).split(':'))
        return (h * 3600 + m * 60 + s) * 32000

    def __len__(self):
        return len(self.df_real)

    def __getitem__(self, idx):
        # Get rows
        row_real = self.df_real.iloc[idx]
        row_teacher = self.df_teacher.iloc[idx]

        # Create path to load file
        filename = row_real["row_id"]
        path_filename = os.path.join(self.dir_soundscapes, filename)

        # Get time to load audio chunk
        start = self.time_to_sec(row_real["start"])
        end = self.time_to_sec(row_real["end"])

        # Load waveform
        chunk, sample_rate = sf.read(path_filename, start=start, stop=end)
        chunk = torch.tensor(chunk, dtype=torch.float32)

        # Load real target
        y_real = row_real.iloc[3:].to_numpy(dtype=np.float32)
        y_real = torch.tensor(y_real, dtype=torch.float32)

        # Load teacher soft target
        y_teacher = row_teacher.iloc[3:].to_numpy(dtype=np.float32)
        y_teacher = torch.tensor(y_teacher, dtype=torch.float32)

        return chunk, y_real, y_teacher