import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import ast
import os
from utils.dictionary import DICTIONARY_LABEL2TARGET, DICTIONARY_LABEL2WEIGHT

class UtilFunctions():
    '''
    Utility class for handling label preprocessing in multi-label
    audio classification tasks.

    Includes:
    - Multi-hot encoding generation
    - Label aggregation for soundscapes
    - Label smoothing for regularization
    '''

    def __init__(self):
        # Initialize dictionaries and number of classes
        self.label2target = DICTIONARY_LABEL2TARGET
        self.label2weight = DICTIONARY_LABEL2WEIGHT
        self.n_classes = len(self.label2target)

    # Create multihot target combining primary and secondary labels from train.csv
    def multi_hot_audios(self, primary_label, secondary_labels = "[]"):
        vec_1 = self.label2target[primary_label].copy()  # Start from primary label one-hot vector
        if secondary_labels != "[]": 
            # If secondary labels are not empty then add the corresponding target
            s_labels = ast.literal_eval(secondary_labels)  # into Python list
            for label in s_labels:
                vec_2 = self.label2target[label]
                vec_1 = np.maximum(vec_1, vec_2)
        return  vec_1
        
    # Create multihot target from soundscapes dataset primary labels
    def multi_hot_soundscapes(self, primary_label):
        p_labels = primary_label.split(";")  # Split multiple labels separated by ';'
        vec_1 = np.zeros(self.n_classes)   # Initialize empty multi-hot vector
        for label in p_labels:
            # Aggregate all label vectors into a single multi-hot vector
            vec_2 = self.label2target[label]
            vec_1 = np.maximum(vec_1, vec_2)
        return vec_1
    
    # LABEL SMOOTHING (For knowledge distilation)
    def label_smooth(self, y_true, mu = 0.05):
        y_true_smooth = y_true * (1 - mu)  +  (mu * torch.sum(y_true, dim=1, keepdim=True) / self.n_classes) 
        return y_true_smooth
    
    # Function to calculate ROC-AUC as in BirdsCLEF for one class
    def one_class_auc(self, y_true, y_pred):

        # Transform to numerical
        y_true = y_true.astype(float).astype(int)
        y_pred = y_pred.astype(float)

        # Mask to select true negatives and positives
        positives  = y_pred[y_true==1]
        negatives = y_pred[y_true==0]
        total_pairs = len(positives) * len(negatives)
        if total_pairs == 0:
            return np.nan
        
        # Calculate AUC
        count = 0
        for p in positives:
            count += (p > negatives).sum()
            count += 0.5 * (p == negatives).sum()
        auc = count/total_pairs
        return auc


class LossCombined(nn.Module):
    """Multilabel loss: combination of BCE and Focal Loss."""
    
    def __init__(self, beta = 0.3, rho = 2):
        super().__init__()
        self.beta = beta  # If beta=0.3: final_loss = 0.3 * BCE + 0.7 * Focal
        self.rho = rho    
        self.bce = nn.BCELoss(reduction = "none")   # reduction="none" -> do NOT average yet, we want per-element loss
        self.epsilon = 1e-7

    def loss_focal(self, y_pred, y_true):
        val_1 = y_true * ( (1 - y_pred) ** self.rho ) * torch.log(y_pred)
        val_0 = (1 - y_true) * ( y_pred ** self.rho ) * torch.log(1 - y_pred)
        sum = -val_1 - val_0
        return sum

    def forward(self, y_logits, y_true):
        
        # Transform logit to prob
        y_pred = torch.sigmoid(y_logits)
        y_pred = torch.clamp(y_pred, self.epsilon, 1-self.epsilon)

        # calculate BCE elementwise
        bce_loss = self.bce(y_pred, y_true)

        # calculate focal elementwise
        focal_loss = self.loss_focal(y_pred, y_true)

        # Combine both functions
        loss = self.beta * bce_loss + ((1-self.beta) * focal_loss)

        return loss.mean()




if __name__ == "__main__":
    # Instancia de la clase
    md = UtilFunctions()

    # ---- Test 1: multi_hot_audios ----
    # Ajusta estos labels a los que realmente existan en tu diccionario
    primary = list(md.label2target.keys())[0]
    secondary = str([list(md.label2target.keys())[1]])  # string tipo "['label2']"

    vec_audio = md.multi_hot_audios(primary, secondary)
    print("Test multi_hot_audios:")
    print("Primary:", primary)
    print("Secondary:", secondary)
    print("Vector:", vec_audio)
    print("Non-zero indices:", np.where(vec_audio == 1)[0])
    print()

    # ---- Test 2: sin secundarios ----
    vec_audio_no_sec = md.multi_hot_audios(primary, "[]")
    print("Test sin secundarios:")
    print("Vector:", vec_audio_no_sec)
    print()

    # ---- Test 3: multi_hot_soundscapes ----
    labels = list(md.label2target.keys())[:3]
    soundscape_input = ";".join(labels)

    vec_soundscape = md.multi_hot_soundscapes(soundscape_input)
    print("Test multi_hot_soundscapes:")
    print("Input:", soundscape_input)
    print("Vector:", vec_soundscape)
    print("Non-zero indices:", np.where(vec_soundscape == 1)[0])