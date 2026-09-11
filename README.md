# Acoustic Leak Detection

Robustness analysis of episodic acoustic leak detection under varying noise conditions.

## 📄 Paper

**Mustapha, A.K.** (2026). *Robustness of Episodic Acoustic Leak Detection Across Noise Levels.* engrXiv. DOI: [10.31224/8122](https://doi.org/10.31224/8122)

## 📊 Dataset

**GPLA-12** — Gas pipeline leak acoustic signal dataset

| Property | Value |
|----------|-------|
| Samples | 780 |
| Classes | 12 leak types |
| Sampling rate | 1000 Hz |
| Signal length | 888 samples (0.888 s) |
| Split | 60% train (468), 20% val (156), 20% test (156) |

Source: [GPLA-12 on arXiv](https://arxiv.org/abs/2106.10277)

## 🏗️ Architecture

- **Conformer encoder** — combines CNN (local features) with self-attention (global context)
- **Prototypical network** — metric-based few-shot classification
- **Episodic training** — 5-shot, N-way episodes
- **Model size** — ~3.2M parameters

## 📈 Results

Validation accuracy across SNR levels (mean ± std over 5 seeds):

| SNR | Mean Accuracy | Δ vs Clean |
|-----|---------------|------------|
| Clean | 65.88% ± 1.36% | — |
| 20 dB | 66.31% ± 0.97% | +0.43% |
| 15 dB | 67.41% ± 0.92% | +1.53% |
| **10 dB** | **68.04% ± 0.66%** | **+2.16%** (p < 0.05) |
| 5 dB | 67.73% ± 0.15% | +1.85% |
| 0 dB | 63.07% ± 1.30% | −2.81% (p < 0.05) |

**Key finding**: Moderate noise (10 dB SNR) improves accuracy via a regularization effect, while extreme noise (0 dB) degrades performance.

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/K-Musty/acoustic-leak-detection.git
cd acoustic-leak-detection

# Install dependencies
pip install torch torchaudio numpy pandas matplotlib scikit-learn librosa tqdm pyyaml

# Run baseline training
PYTHONPATH=. python src/train.py configs/baseline.yaml

# Run noise-robustness experiments
PYTHONPATH=. python src/train.py configs/noise_10db.yaml
PYTHONPATH=. python src/train.py configs/noise_0db.yaml
```

### Configuration

All experiments are defined in `configs/`. Key parameters:

| Parameter | Value |
|-----------|-------|
| Shot | 5 |
| Episodes per epoch | 300 |
| Learning rate | 1e-4 |
| Embedding dim | 128 |
| Conformer layers | 4 |
| Attention heads | 4 |
| Kernel size | 31 |

## 📁 Repository Structure

```
acoustic-leak-detection/
├── configs/           # Experiment configurations
│   ├── baseline.yaml
│   ├── noise_0db.yaml
│   ├── noise_5db.yaml
│   ├── noise_10db.yaml
│   ├── noise_15db.yaml
│   ├── noise_20db.yaml
│   └── noise_clean.yaml
├── src/
│   ├── data/          # Dataset loading
│   ├── models/
│   │   ├── conformer_encoder.py
│   │   └── prototypical.py
│   └── train.py       # Training entry point
├── notebooks/         # Exploration notebooks
├── data/              # Data (not tracked)
├── experiments/       # Results (not tracked)
└── paper/             # Manuscript files
```

## 📊 Data Access

The GPLA-12 dataset is publicly available:

- **Paper**: [arXiv:2106.10277](https://arxiv.org/abs/2106.10277)
- Download the dataset and place it in `data/raw/`
- Run preprocessing to generate `data/processed/gpla_v2/`

## 🔧 Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (tested on NVIDIA T4 via Google Colab)

## 📝 Notes

- Training accuracy was **not separately logged**; only validation accuracy per epoch is available.
- Results reported in the paper are from 5 random seeds: 42, 123, 456, 789, 101112.
- Statistical significance was assessed using paired t-tests across seeds.

## 📧 Contact

**Abdulrahman Kalli Mustapha**
- Email: kmustapha9564@gmail.com
- Affiliation: Al-Qalam University, Katsina, Nigeria
- GitHub: [@K-Musty](https://github.com/K-Musty)

## 📄 License

MIT License
