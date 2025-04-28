# Latent Space Alignment for AI-Native MIMO Semantic Communications

> [!TIP]
> Semantic communications focus on prioritizing the understanding of the meaning behind transmitted data and ensuring the successful completion of tasks that motivate the exchange of information. However, when devices rely on different languages, logic, or internal representations, semantic mismatches may occur, potentially hindering mutual understanding. This paper introduces a novel approach to addressing latent space misalignment in semantic communications, exploiting multiple-input multiple-output (MIMO) communications. Specifically, our method learns a MIMO precoder/decoder pair that jointly performs latent space compression and semantic channel equalization, mitigating both semantic mismatches and physical channel impairments. We explore two solutions: (i) a linear model, optimized by solving a biconvex optimization problem via the alternating direction method of multipliers (ADMM); (ii) a neural network-based model, which learns semantic MIMO precoder/decoder under transmission power budget and complexity constraints. Numerical results demonstrate the effectiveness of the proposed approach in a goal-oriented semantic communication scenario, illustrating the main trade-offs between accuracy, communication burden, and complexity of the solutions.

## Simulations

This section provides the necessary commands to run the simulations required for the experiments. The commands execute different training scripts with specific configurations. Each simulation subsection contains both the `python` command and `uv` counterpart.

### Accuracy Vs Compression Factor

```bash
# Neural Semantic Precoding/Decoding Aware
python scripts/train_neural.py communication.snr=20.0 seed=27,42,100,123,144,200 communication.antennas_receiver=1,2,4,8,12,24,48,96,192 communication.antennas_transmitter=1,2,4,8,12,24,48,96,192 communication.awareness=aware datamodule.train_label_size=4200,2100,420,210,42,21 simulation=compr_fact -m

# Linear Semantic Precoding/Decoding Aware
python scripts/train_linear.py communication.snr=20.0 seed=27,42,100,123,144,200 communication.antennas_receiver=1,2,4,8,12,24,48,96,192 communication.antennas_transmitter=1,2,4,8,12,24,48,96,192 communication.awareness=aware datamodule.train_label_size=4200,2100,420,210,42,21 simulation=compr_fact -m

# Neural Semantic Precoding/Decoding Unaware
python scripts/train_neural.py communication.snr=20.0 seed=27,42,100,123,144,200 communication.antennas_receiver=1,2,4,8,12,24,48,96,192 communication.antennas_transmitter=1,2,4,8,12,24,48,96,192 communication.awareness=unaware datamodule.train_label_size=2100 simulation=compr_fact -m

# Linear Semantic Precoding/Decoding Unaware
python scripts/train_linear.py communication.snr=20.0 seed=27,42,100,123,144,200 communication.antennas_receiver=1,2,4,8,12,24,48,96,192 communication.antennas_transmitter=1,2,4,8,12,24,48,96,192 communication.awareness=unaware datamodule.train_label_size=2100 simulation=compr_fact -m
```

```bash
# Neural Semantic Precoding/Decoding Aware
uv run scripts/train_neural.py communication.snr=20.0 seed=27,42,100,123,144,200 communication.antennas_receiver=1,2,4,8,12,24,48,96,192 communication.antennas_transmitter=1,2,4,8,12,24,48,96,192 communication.awareness=aware datamodule.train_label_size=4200,2100,420,210,42,21 simulation=compr_fact -m

# Linear Semantic Precoding/Decoding Aware
uv run scripts/train_linear.py communication.snr=20.0 seed=27,42,100,123,144,200 communication.antennas_receiver=1,2,4,8,12,24,48,96,192 communication.antennas_transmitter=1,2,4,8,12,24,48,96,192 communication.awareness=aware datamodule.train_label_size=4200,2100,420,210,42,21 simulation=compr_fact -m

# Neural Semantic Precoding/Decoding Unaware
uv run scripts/train_neural.py communication.snr=20.0 seed=27,42,100,123,144,200 communication.antennas_receiver=1,2,4,8,12,24,48,96,192 communication.antennas_transmitter=1,2,4,8,12,24,48,96,192 communication.awareness=unaware datamodule.train_label_size=2100 simulation=compr_fact -m

# Linear Semantic Precoding/Decoding Unaware
uv run scripts/train_linear.py communication.snr=20.0 seed=27,42,100,123,144,200 communication.antennas_receiver=1,2,4,8,12,24,48,96,192 communication.antennas_transmitter=1,2,4,8,12,24,48,96,192 communication.awareness=unaware datamodule.train_label_size=2100 simulation=compr_fact -m
```

### Accuracy Vs SNR

```bash
# Neural Semantic Precoding/Decoding
python scripts/train_neural.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 communication.antennas_receiver=8 communication.antennas_transmitter=8 datamodule.train_label_size=2100 simulation=snr -m

# Linear Semantic Precoding/Decoding
python scripts/train_linear.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 communication.antennas_receiver=8 communication.antennas_transmitter=8 datamodule.train_label_size=2100 simulation=snr -m

# Baseline First-K
python scripts/train_baseline.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 strategy=First-K communication.antennas_receiver=8 communication.antennas_transmitter=8 datamodule.train_label_size=2100 simulation=snr -m

# Baseline Top-K
python scripts/train_baseline.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 strategy=Top-K communication.antennas_receiver=4 communication.antennas_transmitter=4 datamodule.train_label_size=2100 simulation=snr -m

# Baseline Eigen-K
python scripts/train_baseline.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 strategy=Eigen-K communication.antennas_receiver=8 communication.antennas_transmitter=8 datamodule.train_label_size=2100 simulation=snr -m
```

```bash
# Neural Semantic Precoding/Decoding
uv run scripts/train_neural.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 communication.antennas_receiver=8 communication.antennas_transmitter=8 datamodule.train_label_size=2100 simulation=snr -m

# Linear Semantic Precoding/Decoding
uv run scripts/train_linear.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 communication.antennas_receiver=8 communication.antennas_transmitter=8 datamodule.train_label_size=2100 simulation=snr -m

# Baseline First-K
uv run scripts/train_baseline.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 strategy=First-K communication.antennas_receiver=8 communication.antennas_transmitter=8 datamodule.train_label_size=2100 simulation=snr -m

# Baseline Top-K
uv run scripts/train_baseline.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 strategy=Top-K communication.antennas_receiver=4 communication.antennas_transmitter=4 datamodule.train_label_size=2100 simulation=snr -m

# Baseline Eigen-K
uv run scripts/train_baseline.py communication.snr=-20.0,-10.0,10.0,20.0,30.0 seed=27,42,100,123,144,200 strategy=Eigen-K communication.antennas_receiver=8 communication.antennas_transmitter=8 datamodule.train_label_size=2100 simulation=snr -m
```

### Classifiers

The following command will initiate training of the required classifiers for the above simulations. However, this step is not strictly necessary, as the simulation scripts will automatically check for the presence of pretrained classifiers in the `models/classifiers` subfolder. If the classifiers are not found, a pretrained version (used in our paper) will be downloaded from Drive.

```bash
# Classifiers
python scripts/train_classifier.py seed=27,42,100,123,144,200 -m
```

```bash
# Classifiers
uv run scripts/train_classifier.py seed=27,42,100,123,144,200 -m
```

## Dependencies  

### Using `pip` package manager  

It is highly recommended to create a Python virtual environment before installing dependencies. In a terminal, navigate to the root folder and run:  

```bash
python -m venv <venv_name>
```

Activate the environment:  

- On macOS/Linux:  

  ```bash
  source <venv_name>/bin/activate
  ```

- On Windows:  

  ```bash
  <venv_name>\Scripts\activate
  ```

Once the virtual environment is active, install the dependencies:  

```bash
pip install -r requirements.txt
```

You're ready to go! 🚀  

### Using `uv` package manager (Highly Recommended)  

[`uv`](https://github.com/astral-sh/uv) is a modern Python package manager that is significantly faster than `pip`.  

#### Install `uv`  

To install `uv`, follow the instructions from the [official installation guide](https://github.com/astral-sh/uv#installation).  

#### Set up the environment and install dependencies  

Run the following command in the root folder:  

```bash
uv sync
```

This will automatically create a virtual environment (if none exists) and install all dependencies.  

You're ready to go! 🚀  

## Authors

- [Mario Edoardo Pandolfo](https://github.com/JRhin)
- [Simone Fiorellino](https://scholar.google.com/citations?hl=en&user=nKMc4GQAAAAJ)
- [Paolo Di Lorenzo](https://scholar.google.com/citations?hl=en&user=VZYvspQAAAAJ)
- [Emilio Calvanese Strinati](https://scholar.google.com/citations?user=bWndGhQAAAAJ)

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
