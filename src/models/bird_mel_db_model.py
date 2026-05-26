import torch
import torch.nn as nn
import random
import timm
import torchaudio

class BirdEfficientNetV2S(nn.Module):
    def __init__(self, num_classes=234, pretrained=True, dropout=0.25, n_fft=2048, hop_length=512, n_mels=256, p_mask = 0.5):

        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        self.p_mask = p_mask

        self.global_mean_mel_db = -5.6946048623856464  # with 5s segments audios
        self.global_std_mel_db = 23.86795486727926     # with 5s segments audios
        
        # CNN backbone
        self.backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k", 
                        pretrained=pretrained, in_chans=1, num_classes=0, global_pool="avg")

        # Audio to Log-Mel
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=32000, 
                        n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
        self.db = torchaudio.transforms.AmplitudeToDB()

        # SpecAugment masking
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=50)

        # Classification head
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes)
        )

    def forward(self, x):
        """
        Args = x: input tensor of shape [B, 32kHz * 5s]

        Returns =  y: raw class scores of shape [B, num_classes]
        """
        # Transform waveform to log-mel
        x = self.mel(x)  # [B,128,T]
        x = self.db(x)  

        #SpecAugment with prob. p_mask
        if self.training:
            if random.random() < self.p_mask:
                x = self.freq_mask(x)
                x = self.time_mask(x)

        # Normalize
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)  
        #mean = self.global_mean_mel_db
        #std = self.global_std_mel_db    

        x = (x - mean) / (std + 1e-6)

        # CNN
        x = x.unsqueeze(1)  # [B,1,128,T]
        x = self.backbone(x)  # [B,feat_dim]
        # Head
        y = self.head(x)  # [B,234]

        return y
    

