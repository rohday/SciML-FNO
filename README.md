# 2D Heterogenous Poisson ML solver using Fourier Neural Operators

Fourier Neural Operator for solving the 2D heterogeneous Poisson equation.

## Quick Start

```bash
# Generate data
cd data_generation
python3 generate_dataset.py --train_samples 1000 --val_samples 200 --test_samples 200 --output ../data/

# Train
python3 -m model.train --data_path data/train.npz --epochs 300 --output_dir checkpoints/ --learning_rate 1e-2

# Evaluate (this happens on its own during training)
python3 -m model.evaluate --checkpoint checkpoints/ --data_path data/test.npz
```

## Structure

```
├── model/           # FNO implementation, training, evaluation
├── data_generation/ # Poisson solver and dataset generation
├── benchmarks/      # Throughput stress tests
└── allmodels/       # Pre-trained models
```

## Requirements

```
torch numpy scipy matplotlib tqdm h5py pyyaml psutil
```
```
contourpy==1.3.3
cuda-bindings==12.9.4
cuda-pathfinder==1.3.3
cycler==0.12.1
filelock==3.20.3
fonttools==4.61.1
fsspec==2026.1.0
h5py==3.15.1
Jinja2==3.1.6
kiwisolver==1.4.9
MarkupSafe==3.0.3
matplotlib==3.10.8
mpmath==1.3.0
networkx==3.6.1
numpy==2.4.1
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.5
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvshmem-cu12==3.4.5
nvidia-nvtx-cu12==12.8.90
packaging==26.0
pillow==12.1.0
psutil==7.2.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
PyYAML==6.0.3
scipy==1.17.0
setuptools==80.10.2
six==1.17.0
sympy==1.14.0
torch==2.10.0
tqdm==4.67.1
triton==3.6.0
typing_extensions==4.15.0
```

