import torch
import yaml
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from box import Box
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets.dataset_teach import BirdTeachData
from models.bird_mel_db_model import BirdEfficientNetV2S
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
path = os.path.expanduser("./teaching/df_teach_predictions.csv")
df_teach = pd.read_csv(path)
data_train = BirdTeachData(df = df_teach)

# Create Dataloader 
dataloader = DataLoader(data_train, batch_size=val.train.batch_size, shuffle=True, num_workers=val.train.num_workers, 
                        pin_memory=val.train.pin_memory, persistent_workers=val.train.persistent_workers, 
                        prefetch_factor=val.train.prefetch_factor)

# Model 
model = BirdEfficientNetV2S(dropout=0.15, n_fft=2048, hop_length=512, n_mels=128).to(device)

# Loss + Optimizer 
criterion = LossCombined(beta = val.loss.beta, rho = val.loss.rho) # If beta=0.3: final_loss = 0.3 * BCE + 0.7 * Focal
optimizer = torch.optim.AdamW(model.parameters(), lr=val.train.lr, weight_decay=val.train.wd)


# Load scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=val.train.n_epochs, eta_min=1e-6)


# Print config
print(f"""
Training configuration
----------------------
Total Epochs        : {val.train.n_epochs}
Batch size          : {val.train.batch_size}
Learning rate       : {val.train.lr}
Weight decay        : {val.train.wd}

# Mel Configuration
----------------------
Window size         : {val.spectrogram.n_fft}
Frame hop / Stride  : {val.spectrogram.hop_length}
Mel bins            : {val.spectrogram.n_mels}

Augmentation configuration
----------------------
Mel-Mask probability: {val.augment.p_mask}
Noise probability   : {val.augment.p_noise}
Noise alpha         : {val.augment.alpha}
MixUp probability   : {val.augment.p_mix}
Filter probability  : {val.augment.p_filter}

Loss and other Parameters
----------------------
Loss BCE factor        : {val.loss.beta}
Loss Focal factor      : {1.0 - val.loss.beta}
Focusing factor        : {val.loss.rho}
Data Sampling Gamma    : {val.augment.gamma}
""")


# Name model
model_name = f"model{val.train.n_epochs}_{val.model.m_type}_student.pt"



# ============================ TRAINING LOOP =============================  
train_losses = []
for epoch in range(val.train.n_epochs):
    # Put model to train 
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{val.train.n_epochs}") # Create progress bar
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
                "scheduler_state_dict": scheduler.state_dict(), "train_losses": train_losses}
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