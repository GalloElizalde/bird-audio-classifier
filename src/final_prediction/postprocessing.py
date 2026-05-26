import pandas as pd
import os
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

class predict():
    def __init__(self, filename, model, mode : str = "mean", n = 1):
        self.path_test = "./test_soundscapes/"
        self.filename = filename
        self.sample_rate = 32000
        self.mode = mode
        self.model = model
        self.n = n

    def get_audio_prediction(self):
        # Load waveform
        path_wave = os.path.join(self.path_test, self.filename)
        waveform, _ = sf.read(path_wave)
        segment_samples = int(5 * self.sample_rate)
        # Cut waveform in 12 chunks of 5s each and save name and prediction
        predictions = []
        filenames = []
        self.model.eval()
        for i in range(12):
            # audio to pred
            print(i)        
            start =  i * segment_samples
            end = start + segment_samples
            chunk = waveform[start:end]
            y_pred = self.model.predict(chunk)
            predictions.append(y_pred)

            # Filename
            end = "_" + str((i + 1) * 5)
            f_name = self.filename.replace(".ogg", "") + end
            filenames.append(f_name)
        return filenames, torch.stack(predictions)
    
    def mean_method(self, predictions):
        predictions_mean  = torch.mean(predictions, dim=0)
        new_pred = predictions * predictions_mean
        return new_pred

    def top_n_method(self, predictions):
        vals, _ = torch.topk(predictions, k=self.n, dim=0)
        vals_mean = torch.mean(vals, dim = 0)
        new_pred = predictions * vals_mean
        return new_pred

    def conv_method(self, predictions):
    # predictions: [12, 234]
        x = predictions.T.unsqueeze(0)
        kernel = torch.tensor([0.25, 0.5, 0.25], dtype=x.dtype, device=x.device).view(1, 1, 3)
        kernel = kernel.repeat(predictions.shape[1], 1, 1)
        out = F.conv1d(x, kernel, padding=1, groups=predictions.shape[1])
        return out.squeeze(0).T
        
    def get_final_prediction(self, n):
        # Get final predictions and filenames
        name2pred = {}
        file_names, predictions  = self.get_audio_prediction()
        if self.mode == "mean":
            new_predictions = self.mean_method(predictions)
        elif self.mode == "top_n":
            new_predictions = self.top_n_method(predictions)
        for name, pred in zip(file_names, new_predictions):
            name2pred[name] = pred
        return name2pred


