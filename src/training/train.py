import torch
import yaml
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from box import Box
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets.dataset_audios import BirdAudioDataset
from models.bird_mel_db_model import BirdEfficientNetV2S
from models.bird_wigner_model import BirdWignerNet
from utils.util_functions import LossCombined

# Load config yaml
with open("./config/config.yaml", "r") as f:
    val = Box(yaml.safe_load(f))

# Optimization and select device
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision(val.debug.matmul_precision)  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataframe 
path = os.path.expanduser("~/1_machine_learning_projects/BIRDS/BIRDS/src/data/df_with_folds.csv")
df_train = pd.read_csv(path)
data_train = BirdAudioDataset(df = df_train, p_noise = val.augment.p_noise, p_filter = val.augment.p_filter, 
                              p_mix = val.augment.p_mix, p_soundscape_noise = val.augment.p_soundscape_noise, alpha = val.augment.alpha)


# Create Dataloader using sampler w load weights
sample_weights = torch.tensor(df_train["sample_weight"].values ** val.augment.gamma, dtype=torch.double)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
dataloader = DataLoader(data_train, batch_size=val.train.batch_size, sampler=sampler, 
                        shuffle=False, num_workers=val.train.num_workers, pin_memory=val.train.pin_memory, 
                        persistent_workers=val.train.persistent_workers, prefetch_factor=val.train.prefetch_factor)

# Model
if val.model.m_type == "dbmel":
    model = BirdEfficientNetV2S(dropout=val.model.dropout, n_fft=val.spectrogram.n_fft, 
        hop_length=val.spectrogram.hop_length, n_mels=val.spectrogram.n_mels, p_mask = val.augment.p_mask).to(device)
elif val.model.m_type == "wigner":
    model = BirdWignerNet(dropout=val.model.dropout, n_fft=val.spectrogram.n_fft, hop_length=val.spectrogram.hop_length).to(device)

# Loss + Optimizer 
criterion = LossCombined(beta = val.loss.beta, rho = val.loss.rho) # If beta=0.3: final_loss = 0.3 * BCE + 0.7 * Focal
optimizer = torch.optim.AdamW(model.parameters(), lr=val.train.lr, weight_decay=val.train.wd)


# Check if new training or expand training with checkpoints parameters
start_epoch = 0
checkpoint_path = f"output/{val.train.resume_training_model}.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    train_losses = checkpoint["train_losses"]
    start_epoch = checkpoint["epoch"]
    total_epochs = start_epoch + val.train.n_epochs
    version = checkpoint["version"] + 1
    print(f"Resuming from epoch {start_epoch} to {total_epochs}\n")
else:
    print(f"First training\n") 
    total_epochs = start_epoch + val.train.n_epochs
    train_losses = []
    version = 1  

# Load scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)
if os.path.exists(checkpoint_path):
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

# Print config
print(f"""
# Training configuration
----------------------
Model Type:         : {val.model.m_type}
Total Epochs        : {total_epochs}
Starting Epoch      : {start_epoch}
Batch size          : {val.train.batch_size}
Learning rate       : {val.train.lr}
Weight decay        : {val.train.wd}

# Mel Configuration
----------------------
Window size         : {val.spectrogram.n_fft}
Frame hop / Stride  : {val.spectrogram.hop_length}
Mel bins            : {val.spectrogram.n_mels}

# Augmentation configuration
----------------------
Mel-Mask probability: {val.augment.p_mask}
Noise probability   : {val.augment.p_noise}
Noise alpha         : {val.augment.alpha}
MixUp probability   : {val.augment.p_mix}
Filter probability  : {val.augment.p_filter}

# Loss and other Parameters
----------------------
Loss BCE factor        : {val.loss.beta}
Loss Focal factor      : {1.0 - val.loss.beta}
Focusing factor        : {val.loss.rho}
Data Sampling Gamma    : {val.augment.gamma}
""")


# Name model
mel_name = f"nfft{val.spectrogram.n_fft}_hop{val.spectrogram.hop_length}_nmels{val.spectrogram.n_mels}"
mel_name = mel_name.replace(".", "p")

param_name = f"gamma{val.augment.gamma}_beta{val.loss.beta}_rho{val.loss.rho}"
param_name = param_name.replace(".","p")
param_name = param_name.replace("-","m")

prob_name = f"noise{val.augment.p_noise}_mix{val.augment.p_mix}_filter{val.augment.p_filter}_mask{val.augment.p_mask}"
prob_name = prob_name.replace(".","p")

if val.model.m_type == "dbmel":
    model_name = f"model{total_epochs}_{val.model.m_type}_{param_name}_{mel_name}_{prob_name}_v{version}.pt"
else:
    model_name = f"model{total_epochs}_{val.model.m_type}_{param_name}_{prob_name}_v{version}.pt"


# ============================ TRAINING LOOP =============================  
for epoch in range(start_epoch, total_epochs):
    # Put model to train 
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{total_epochs}") # Create progress bar
    for batch_X, batch_y in pbar:
        # To cpu
        batch_X = batch_X.to(device, non_blocking=True)
        batch_y = batch_y.float().to(device, non_blocking=True)
        # Reinitialize gradients
        optimizer.zero_grad(set_to_none=True)
        # Forward
        logits = model(batch_X)
        # Loss
        loss = criterion(logits, batch_y)
        epoch_loss += loss.item()
        # Backprop
        loss.backward()
        optimizer.step()
    # Track loss
    epoch_loss /= len(dataloader)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1} | Loss: {epoch_loss:.4f}")
    # Scheduler
    scheduler.step()
  
# Save checkpoint
checkpoint = {"epoch": epoch + 1, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), 
                "scheduler_state_dict": scheduler.state_dict(), "train_losses": train_losses, "version": version}
torch.save(checkpoint, f"output/{model_name}")
print(f"Checkpoint saved at output/{model_name}")


# Save Loss Plot
name_plot = model_name.replace(".pt", "")
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.grid(True)
plt.savefig(f"./output/loss_curve_{name_plot}.png", dpi = 150)
plt.close()