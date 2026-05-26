# Bird Audio Classifier

## Overview

The goal of this project is to investigate whether Wigner-based time-frequency representations, originally motivated by concepts from quantum mechanics, can provide advantages over conventional log-mel spectrograms in CNN-based audio classification systems.

---

## Description

In particular, this project investigates whether the Wigner distribution can improve discriminative performance compared to conventional log-mel spectrograms in the context of bird audio multi-label classification.

Using data from the BirdCLEF+ 2026 challenge and inspired by the BirdCLEF+ 2025 benchmark setup proposed by Kahl et al. (2025), this is explored using a PyTorch framework with EfficientNetV2 backbones trained on bioacoustic soundscape data.

The project includes:

- Short-Time Fourier Transform (STFT) representations.
- Log-mel spectrogram representations.
- Wigner-Ville time-frequency distributions.
- Audio-domain augmentation techniques, including masking, waveform mixing, noise injection, and random filtering.
- Data sampling strategies and combined loss formulations (Focal Loss + Binary Cross Entropy) to mitigate severe class imbalance.
- Pseudo-labeling and teacher-student distillation.
- Model validation using 5-fold cross-validation.
- Model ensembling for more robust inference under noisy environmental conditions.
- Post-processing techniques, including label smoothing and probability reweighting.

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

### Log-Mel Baseline

The main baseline converts raw waveforms into log-mel spectrograms using `torchaudio` transforms and feeds them into an EfficientNetV2-S CNN backbone.

#### Pipeline

```text
Waveform
   ↓
MelSpectrogram
   ↓
AmplitudeToDB
   ↓
SpecAugment
   ↓
EfficientNetV2-S
   ↓
Multilabel classifier
```

Implemented in:

```text
models/bird_mel_db_model.py
```

---

### Wigner Distribution Variant

A second model explores the Wigner-Ville distribution as a richer time-frequency representation.

#### Pipeline

```text
Waveform
   ↓
Analytic signal (Hilbert transform)
   ↓
Wigner distribution
   ↓
Log compression
   ↓
EfficientNetV2-S
   ↓
Multilabel classifier
```

Implemented in:

```text
models/bird_wigner_model.py
```

---

The project uses a centralized `config.yaml` file to control model behavior, spectrogram generation, augmentation strategies, and loss balancing. Besides standard training and testing parameters, the configuration system allows rapid experimentation with different time-frequency representations and augmentation policies.

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

Training samples are built from:

- bird recordings,
- environmental soundscapes,
- ESC-50 noise sources.

Implemented augmentations include:

- random noise injection,
- random EQ filtering,
- MixUp-style waveform mixing,
- SpecAugment masking.

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

- **Mel-Mask probability** controls SpecAugment masking probability.
- **Noise probability** determines the probability of random noise injection.
- **Noise alpha** controls injected noise magnitude.
- **MixUp probability** enables waveform mixing augmentation.
- **Filter probability** applies random equalization filtering.

---

## Loss and Other Parameters

```text
Loss BCE factor        : 0.3
Loss Focal factor      : 0.7
Focusing factor        : 2.0
Data Sampling Gamma    : -0.5
```

- **Loss BCE factor** controls Binary Cross Entropy contribution.
- **Loss Focal factor** controls Focal Loss contribution.
- $$
L = \beta L_{\mathrm{BCE}} + (1 - \beta)L_{\mathrm{Focal}}
$$
- **Focusing factor** adjusts the focal penalty on difficult samples.
- **Data Sampling Gamma** controls weighted sampling behavior for class imbalance mitigation.

---

---

### Evaluation

Evaluation is performed on timestamped soundscape chunks with multilabel targets.

Implementation:

```text
datasets/dataset_soundscapes.py
```

---

## Training Strategy

The project includes:

- weighted random sampling,
- cosine annealing learning-rate scheduling,
- teacher-student distillation,
- multi-model ensembling.

Main training entrypoints:

```text
train.py
teach.py
ensemble_predict.py
```

---

## Results

Best student model achieved approximately:

$$
\text{Macro ROC-AUC} \approx 0.80
$$

with 8-model ensemble as teacher with different parameters of n_fft, n_mels and 
Main observations:

- log-mel representations consistently outperformed Wigner-based representations,
- Ensemble averaging improved stability,
- Augmentation improved generalization,
- Wigner representations increased computational cost without measurable AUC gains in the current setup.

---

## Repository Structure

```text
## Repository Structure

```text
bird-audio-classifier/
├── notebooks/
│   ├── audio_exploration.ipynb
│   └── data_analysis.ipynb
│
├── src/
│   ├── config/
│   │   └── config.yaml
│   │
│   ├── datasets/
│   │   ├── dataset_audios.py
│   │   ├── dataset_soundscapes.py
│   │   └── dataset_teach.py
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
│   ├── teaching/
│   │   ├── teach.py
│   │   └── ensemble_predict.py
│   │
│   ├── testing/
│   │   └── test.py
│   │
│   └── utils/
│       ├── dictionary.py
│       └── util_functions.py
```

---

## Future Directions

Potential future improvements:

- Embedding models,
- Attention pooling.

---

## Disclaimer
## Citation

If you use this repository, please cite:

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

This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

https://creativecommons.org/licenses/by-nc-sa/4.0/

## Disclaimer

This repository is an experimental research framework for multilabel bioacoustic classification, soundscape analysis, and time-frequency representation learning.
