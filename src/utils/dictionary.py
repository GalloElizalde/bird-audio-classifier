import os
import pandas as pd
import numpy as np

# Create a dictionary of one-hot target per class

# load unique lables from taxnonomy csv 
path_csv = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/taxonomy.csv")
df = pd.read_csv(path_csv)
primary_label = df["primary_label"].unique()

# Create label to target dict
n_classes = len(primary_label)
DICTIONARY_LABEL2TARGET = {}
for i, label in enumerate(primary_label):
    vec = np.zeros(n_classes) 
    vec[i] = 1
    DICTIONARY_LABEL2TARGET[label] = vec

# Create a dictionary label to sample weight (sampled by train audio only)
df_audio = pd.read_csv("~/1_machine_learning_projects/BIRDS/data/train.csv")
temp_dict = df_audio["primary_label"].value_counts().to_dict()
total_count = len(df_audio)

DICTIONARY_LABEL2WEIGHT = {}
for label, count in temp_dict.items():
    w = (count / total_count)
    DICTIONARY_LABEL2WEIGHT[label] = w





