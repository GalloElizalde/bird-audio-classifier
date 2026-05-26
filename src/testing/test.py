import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml
from box import Box
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from models.bird_mel_db_model import BirdEfficientNetV2S
from models.bird_wigner_model import BirdWignerNet
from datasets.dataset_soundscapes import BirdSoundscapeDataset 
from utils.util_functions import UtilFunctions
from datetime import datetime
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load config yaml
with open("./config/config.yaml", "r") as f:
    val = Box(yaml.safe_load(f))

# Load test dataset and test DataLoader
path = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_soundscapes_labels.csv")
df = pd.read_csv(path)
data_test = BirdSoundscapeDataset(df)
loader_test = DataLoader(data_test, batch_size=val.test.batch_size, shuffle=False)

# Load model
dir_models = "./output/"
model_name = (val.test.model_name).replace(".pt","")
model_name_vector = (model_name).split("_")
if "dbmel" in model_name_vector:
    n_fft = int(model_name_vector[5].replace("nfft", ""))
    hop_length = int(model_name_vector[6].replace("hop", ""))
    n_mels = int(model_name_vector[7].replace("nmels", ""))
    model = BirdEfficientNetV2S(pretrained=False, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels).to(device)
elif "wigner" in model_name_vector:
    n_fft = int(model_name_vector[5].replace("nfft", ""))
    hop_length = int(model_name_vector[6].replace("hop", ""))
    model = BirdWignerNet(pretrained=False, n_fft=n_fft, hop_length=hop_length).to(device)

# Load checkpoint
checkpoint = torch.load(dir_models + val.test.model_name, map_location=device)

# Load model weights only
model.load_state_dict(checkpoint["model_state_dict"])
#model.load_state_dict(torch.load(dir_models + val.test.model_name, map_location=device))

# Test
all_preds = []
all_targets = []
all_data = []
model.eval()
with torch.no_grad():
    pbar = tqdm(loader_test)
    for batch_X, batch_y, batch_data in pbar:
        # Move tensors to GPU/CPU
        batch_X = batch_X.to(device)
        # Forward pass
        y_pred = model(batch_X)
        # Convert logits -> probabilities
        y_pred = torch.sigmoid(y_pred)
        # Store predictions and targets
        all_preds.append(y_pred.cpu())
        all_targets.append(batch_y.cpu())
        all_data.extend(list(zip(*batch_data)))

# Concatenate all batches
all_preds = torch.cat(all_preds).numpy()
all_targets = torch.cat(all_targets).numpy()

# Combine predictions, id and labels data
path_taxonomy = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/taxonomy.csv") 
df_taxonomy = pd.read_csv(path_taxonomy)
labels = df_taxonomy["primary_label"].values
labels = np.concatenate((["row_id", "start", "end"], labels))
ids = np.array(all_data)[:,0:1]
start = np.array(all_data)[:,1:2]
end = np.array(all_data)[:,2:3]

# Create New dataframes of predictions and targets
predictions = np.hstack((ids, start, end, all_preds))
predictions = pd.DataFrame(predictions)
predictions.columns = labels

targets = np.hstack((ids, start, end, all_targets))
targets = pd.DataFrame(targets)
targets.columns = labels

# Calculate and print Macro ROC-AUC
util = UtilFunctions()
aucs = []
for label in labels[3:]:
    y_true = targets.loc[: ,label]
    y_pred = predictions.loc[:,label]
    auc = util.one_class_auc(y_true, y_pred)
    aucs.append(auc)
aucs = np.array(aucs)
print(f"Macro ROC-AUC: {np.nanmean(aucs):.3f}")

# Save score to scores.csv
df = pd.read_csv("./testing/scores.csv")
day = datetime.now().strftime("%d-%m")
new_row = {"model_name": model_name, "MACRO-ROC-AUC": round(np.nanmean(aucs), 3), "day": day }
df.loc[len(df)] = new_row
df.to_csv("./testing/scores.csv", index=False)