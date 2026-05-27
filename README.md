# Bird Audio Classifier

## Overview

The goal of this project is to investigate whether Wigner-based time-frequency representations, originally motivated by concepts from quantum mechanics, can provide advantages over conventional log-mel spectrograms in CNN-based audio classification systems.

---

## Description

In particular, this project investigates whether the Wigner distribution can improve discriminative performance compared to conventional log-mel spectrograms in the context of bird audio multi-label classification.

Using data from the BirdCLEF+ 2026 challenge and inspired by the BirdCLEF+ 2025 benchmark setup proposed by Kahl et al. (2025), this is explored using a PyTorch framework with EfficientNetV2 backbones trained on bioacoustic soundscape data.

The project includes:

- Log-mel spectrogram representations.
- Wigner-Ville time-frequency distributions.
- Audio-domain augmentation techniques, including masking, waveform mixing, noise injection, and random filtering.
- Data sampling strategies and combined loss formulations (Focal Loss + Binary Cross Entropy) to mitigate severe class imbalance.
- Model ensembling for more robust inference under noisy environmental conditions.
- Post-processing techniques, including probability reweighting methods.
- Transfer learning from isolated bird recordings to soundscape classification.

---

## Research Motivation

Automatic bird recognition from passive audio recordings is challenging due to:

- overlapping vocalizations,
- environmental noise,
- weak labels,
- domain shift between isolated calls and real soundscapes.

Although spectrogram-based representations are commonly used to address these challenges, the Wigner distribution is known to provide a higher joint time-frequency concentration than the spectrogram (Cohen, 1995).

This property also holds in the discrete formulation. While spectrograms smooth the representation through windowing and are constrained by the trade-off between temporal and frequency resolution, the discrete Wigner distribution achieves a sharper concentration of signal energy, allowing a more accurate representation of non-stationary components, chirps, and instantaneous frequencies.

In addition, the Wigner distribution exactly preserves both temporal and spectral energy marginals, making it a more faithful representation of the underlying signal structure.

Its main limitation is the appearance of oscillatory cross-terms in multicomponent signals, which introduce interference artifacts and complicate interpretation. Spectrograms largely avoid this issue due to their inherent smoothing effect.

---
## Architecture Overview

Both models share the same EfficientNetV2-S backbone and multilabel
classification head. The only difference lies in the time-frequency
representation extracted from the input waveform.

### Log-Mel Baseline

```text
waveform → log-mel spectrogram → SpecAugment → EfficientNetV2-S
```

Implementation:
`models/bird_mel_db_model.py`

---

### Wigner Distribution Variant

```text
waveform → analytic signal → Wigner distribution → log compression → EfficientNetV2-S
```

Implementation:
`models/bird_wigner_model.py`

---

The project uses a centralized `config.yaml` file to control all major
training and inference parameters, including:

- model selection,
- spectrogram generation,
- augmentation probabilities,
- optimizer and scheduler parameters,
- weighted sampling behavior,
- BCE/Focal loss balancing,
- checkpoint resume training,
- ensemble and testing configuration.

The configuration system allows rapid experimentation with different
time-frequency representations and training strategies without modifying
the core implementation.

## Mel Configuration

```text
Window size         : 2048
Frame hop / Stride  : 512
Mel bins            : 128
```

- **Window size** controls the FFT analysis window used for spectrogram generation.
- **Frame hop / Stride** determines the temporal spacing between consecutive frames.
- **Mel bins** specifies the number of mel-frequency channels used in the representation.

---

## Data Pipeline

### Audio Dataset

Training samples are constructed from isolated bird recordings.

Each waveform is converted into a fixed-length audio chunk. If the
recording is shorter than the target duration, zero-padding is applied;
otherwise, a random segment is extracted.

The dataset also supports audio-domain augmentation using environmental
soundscapes, ESC-50 noise sources, random EQ filtering, and MixUp-style
waveform mixing.

Targets are represented as multi-hot vectors using both primary and
secondary labels.

Implementation:

```text
datasets/dataset_audios.py
```

## Augmentation Configuration

```text
Mel-Mask probability: 0.1
Noise probability   : 0.5
Noise alpha         : 0.1
MixUp probability   : 0.5
Filter probability  : 0.2
```

- **Mel-Mask probability** controls SpecAugment masking probability during training.
- **Noise probability** controls random environmental noise injection.
- **Noise alpha** controls injected noise amplitude.
- **MixUp probability** controls waveform mixing augmentation.
- **Filter probability** controls random equalization filtering.

---

## Loss and Other Parameters

```text
Loss BCE factor        : 0.3
Loss Focal factor      : 0.7
Focusing factor        : 2.0
Data Sampling Gamma    : -0.5
```

- **Loss BCE factor** controls the Binary Cross Entropy contribution.
- **Loss Focal factor** controls the Focal Loss contribution.

$$
L = \beta L_{\mathrm{BCE}} + (1 - \beta)L_{\mathrm{Focal}}
$$

- **Focusing factor** controls the focal penalty applied to difficult samples.
- **Data Sampling Gamma** controls weighted sampling behavior for class imbalance mitigation.

---


## Training Strategy

Training is performed using a `WeightedRandomSampler` to reduce the
strong class imbalance present in BirdCLEF recordings. Sampling weights
are computed from class frequencies and controlled through the
$\gamma$ parameter.

The training script supports two selectable architectures:

- `BirdEfficientNetV2S`
- `BirdWignerNet`

Model selection is controlled directly from `config.yaml`.

Optimization uses:

- AdamW optimizer,
- cosine annealing learning-rate scheduler,
- combined BCE + Focal multilabel loss.

Training also supports:

- random environmental noise injection,
- MixUp waveform augmentation,
- random EQ filtering,
- SpecAugment masking,
- checkpoint resume training,
- automatic versioned checkpoint saving.

Inference and evaluation are performed using multi-model ensembles with
different spectrogram configurations (`n_fft`, `hop_length`, `n_mels`)
followed by probability-based postprocessing methods.

Main entrypoints:

```text
training/train.py
testing/test.py
```

---

## Results

Best ensemble configuration achieved approximately:

$$
\text{Macro ROC-AUC} \approx 0.80
$$

The final inference pipeline uses an ensemble of multiple
`BirdEfficientNetV2S` models trained with different spectrogram
configurations:

- different `n_fft` values,
- different `hop_length` values,
- different `n_mels` resolutions,
- different BCE/Focal balancing parameters.

For each soundscape segment:

1. logits are converted into probabilities using sigmoid activation,
2. predictions from all ensemble models are averaged,
3. probabilities are postprocessed using:
   - mean-based reweighting,
   - top-N probability reweighting.

Evaluation is performed on timestamped soundscape segments using
Macro ROC-AUC over all bird classes.

Main observations:

- log-mel representations consistently outperformed Wigner-based representations,
- ensemble averaging improved prediction stability,
- probability reweighting improved final ROC-AUC,
- audio-domain augmentation improved generalization,
- Wigner representations increased computational cost without measurable ROC-AUC improvements in the current setup.

---

## Repository Structure

```text
bird-audio-classifier/
├── src/
│   ├── config/
│   │   └── config.yaml
│   │
│   ├── datasets/
│   │   ├── dataset_audios.py
│   │   └── dataset_soundscapes.py
│   │
│   ├── models/
│   │   ├── bird_mel_db_model.py
│   │   └── bird_wigner_model.py
│   │
│   ├── preprocessing/
│   │   └── prepare_data.py
│   │
│   ├── training/
│   │   └── train.py
│   │
│   ├── testing/
│   │   └── test.py
│   │
│   └── utils/
│       ├── dictionary.py
│       └── util_functions.py
│
└── README.md
```

---

## Future Directions

Potential future improvements:

- attention-based pooling strategies,
- embedding-based bioacoustic representations,
- self-supervised audio pretraining,
- improved Wigner cross-term suppression methods,

---

## References

The implementation and experimental setup of this project were mainly
inspired by the following works:

```bibtex
@misc{birdclef-2026,
    author = {Stefan Kahl and Tom Denton and Larissa Sugai and Liliana Piatti and Ryan Holbrook and Holger Klinck and Ashley Oldacre},
    title = {BirdCLEF+ 2026},
    year = {2026},
    howpublished = {\url{https://kaggle.com/competitions/birdclef-2026}},
    note = {Kaggle}
}

@inproceedings{kahl2025birdclef,
  title={Overview of BirdCLEF+ 2025: Multi-Taxonomic Sound Identification in the Middle Magdalena, Colombia},
  author={Kahl, Stefan and Denton, Tom and Sugai, Larissa and others},
  booktitle={CLEF Working Notes},
  year={2025}
}

@book{cohen1995timefrequency,
  title={Time-Frequency Analysis},
  author={Cohen, Leon},
  year={1995},
  publisher={Prentice Hall}
}
```
## License

This project is licensed under the
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
License (CC BY-NC-SA 4.0).

For more information, see:

https://creativecommons.org/licenses/by-nc-sa/4.0/

## Disclaimer

This repository is an experimental research framework for multilabel bioacoustic classification, soundscape analysis, and time-frequency representation learning.
