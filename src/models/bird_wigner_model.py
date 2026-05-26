import torch
import torch.nn as nn
import numpy as np
import timm
from scipy.signal import hilbert

class BirdWignerNet(nn.Module):
    def __init__(self, num_classes=234, pretrained=True, dropout=0.25, n_fft=2048, hop_length=512):
        super().__init__()

        # Wigner parameters
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.global_abs_log1p_mean = 0.14317006690781642
        self.global_abs_log1p_std = 0.3239285445836743

        # CNN backbone
        self.backbone = timm.create_model("tf_efficientnetv2_s.in21k_ft_in1k",
                                          pretrained=pretrained, in_chans=1,
                                          num_classes=0, global_pool="avg")

        # Classification head
        feat_dim = self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(feat_dim, num_classes))

    def wigner_distribution(self, waveform):

        # Transform the real signal into its analytic complex representation
        waveform_np = waveform.detach().cpu().numpy().astype(np.float32)
        s = hilbert(waveform_np)
        s_conj = np.conj(s)
        
        # Pad signal by n_fft//2 samples on both sides
        pad = self.n_fft // 2
        s = np.pad(s, (pad, pad), mode="reflect")  # mode="constant"
        s_conj = np.pad(s_conj, (pad, pad), mode="reflect")

        # Wigner distribution matrix
        m_max = 1 + (len(s) - self.n_fft) // self.hop_length  # Number of complete frames
        W = np.zeros((m_max, self.n_fft), dtype=np.complex64)

        # Loop over time frames
        for m in range(m_max):
            # window boundaries
            start = m * self.hop_length
            end = start + self.n_fft
            frame = s[start:end]

            # Local conjugated segment, reversed to form symmetric lag pairs
            frame_conj = s_conj[start:end]
            frame_conj = frame_conj[::-1]

            # Local symmetric autocorrelation:
            R = frame * frame_conj  # R[n] = s[t+n] s*[t-n]
            W[m, :] = np.fft.fft(R)  # Fourier transform

        # Return W
        W = W[:, :self.n_fft//2]  #Only positive freq.
        W = W.real.astype(np.float32) # Remove small imaginary errors (Wigner is real)
        return torch.from_numpy(W.T)
      

    def forward(self, x):
        """
        Args:
            x: waveform tensor [B, samples]
        Returns:
            y: logits [B, num_classes]
        """

        Ws = []

        for i in range(x.shape[0]):
            W = self.wigner_distribution(x[i])
            W = torch.abs(W)
            W = torch.log1p(W)
            Ws.append(W)

        x = torch.stack(Ws).to(x.device)  # [B, F, T]

        # Per-sample normalization
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        x = (x - mean) / (std + 1e-8)

        x = x.unsqueeze(1)      # [B, 1, F, T]
        x = self.backbone(x)    # [B, feat_dim]
        y = self.head(x)        # [B, 234]

        return y

    


# Example
if __name__ == "__main__":
    waveform = torch.randn(4, 32000 * 5)

    model = BirdWignerNet()

    y = model(waveform)

    print(y)  # [4,234]