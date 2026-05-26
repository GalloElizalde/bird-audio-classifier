from models.bird_mel_db_model import BirdEfficientNetV2S
from datasets.dataset_soundscapes import BirdSoundscapeDataset 
from utils.util_functions import UtilFunctions
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml
from box import Box
from tqdm import tqdm
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load test dataset and test DataLoader
path = os.path.expanduser("~/1_machine_learning_projects/BIRDS/data/train_soundscapes_labels.csv")
df = pd.read_csv(path)
data_test = BirdSoundscapeDataset(df)
loader_test = DataLoader(data_test, batch_size=32, shuffle=False)

# Load models
model_names = [
        "model10_dbmel_gammam0p5_beta0p3_rho2p0_nfft2048_hop512_nmels128_noise0p0_mix0p0_filter0p0_mask0p0_v1.pt",
        "model10_dbmel_gammam0p5_beta0p5_rho2p0_nfft2048_hop512_nmels128_noise0p5_mix0p5_filter0p5_mask0p5_v1.pt",
        "model10_dbmel_gammam0p5_beta1p0_rho2p0_nfft1024_hop320_nmels128_noise0p0_mix0p0_filter0p0_mask0p0_v1.pt",
        "model10_dbmel_gammam0p5_beta1p0_rho2p0_nfft2048_hop512_nmels128_noise0p0_mix0p0_filter0p0_mask0p0_v1.pt",
        "model10_dbmel_gammam0p5_beta1p0_rho2p0_nfft2048_hop512_nmels128_noise0p0_mix0p5_filter0p1_mask0p1_v1.pt",
        "model10_dbmel_gammam0p5_beta1p0_rho2p0_nfft2048_hop512_nmels256_noise0p0_mix0p0_filter0p0_mask0p0_v1.pt"]

dir_models = "./output/"
models = []
for model in model_names:
    model_name = (model).replace(".pt","")
    model_name_vector = (model_name).split("_")
    n_fft = int(model_name_vector[5].replace("nfft", ""))
    hop_length = int(model_name_vector[6].replace("hop", ""))
    n_mels = int(model_name_vector[7].replace("nmels", ""))
    model = BirdEfficientNetV2S(pretrained=False, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels).to(device)
    models.append(model)
model_1 = models[0]
model_2 = models[1]
model_3 = models[2]
model_4 = models[3]
model_5 = models[4]
model_6 = models[5]

# Load and asign checkpoint
checkpoint_1 = torch.load(dir_models + model_names[0], map_location=device)
checkpoint_2 = torch.load(dir_models + model_names[1], map_location=device)
checkpoint_3 = torch.load(dir_models + model_names[2], map_location=device)
checkpoint_4 = torch.load(dir_models + model_names[3], map_location=device)
checkpoint_5 = torch.load(dir_models + model_names[4], map_location=device)
checkpoint_6 = torch.load(dir_models + model_names[5], map_location=device)

model_1.load_state_dict(checkpoint_1["model_state_dict"])
model_2.load_state_dict(checkpoint_2["model_state_dict"])
model_3.load_state_dict(checkpoint_3["model_state_dict"])
model_4.load_state_dict(checkpoint_4["model_state_dict"])
model_5.load_state_dict(checkpoint_5["model_state_dict"])
model_6.load_state_dict(checkpoint_6["model_state_dict"])


# Test
all_preds = []
all_targets = []
all_data = []
model_1.eval()
model_2.eval()
model_3.eval()
model_4.eval()
model_5.eval()
model_6.eval()
with torch.no_grad():
    pbar = tqdm(loader_test)
    for batch_X, batch_y, batch_data in pbar:
        # Move tensors to GPU/CPU
        batch_X = batch_X.to(device)
        # Forward pass, Convert logits -> probabilities
        y_pred_1 = torch.sigmoid(model_1(batch_X))
        y_pred_2 = torch.sigmoid(model_2(batch_X))
        y_pred_3 = torch.sigmoid(model_3(batch_X))
        y_pred_4 = torch.sigmoid(model_4(batch_X))
        y_pred_5 = torch.sigmoid(model_5(batch_X))
        y_pred_6 = torch.sigmoid(model_6(batch_X))

        y_pred = (y_pred_1 + y_pred_2 + y_pred_3 + y_pred_4 + y_pred_5 + y_pred_6) / 6
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
predictions.to_csv("./teaching/teacher_predictions.csv", index = False)

targets = np.hstack((ids, start, end, all_targets))
targets = pd.DataFrame(targets)
targets.columns = labels
targets.to_csv("./teaching/targets.csv", index = False)

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