# Latent Space Alignment for AI-Native MIMO Semantic Communications

## Abstract

Semantic communications focus on prioritizing the understanding of the meaning behind transmitted data and ensuring the successful completion of tasks that motivate the exchange of information. However, when devices rely on different languages, logic, or internal representations, semantic mismatches may occur, potentially hindering mutual understanding. This paper introduces a novel approach to addressing latent space misalignment in semantic communications, exploiting multiple-input multiple-output (MIMO) communications. Specifically, our method learns a MIMO precoder/decoder pair that jointly performs latent space compression and semantic channel equalization, mitigating both semantic mismatches and physical channel impairments. We explore two solutions: (i) a linear model, optimized by solving a biconvex optimization problem via the alternating direction method of multipliers (ADMM); (ii) a neural network-based model, which learns semantic MIMO precoder/decoder under transmission power budget and complexity constraints. Numerical results demonstrate the effectiveness of the proposed approach in a goal-oriented semantic communication scenario, illustrating the main trade-offs between accuracy, communication burden, and complexity of the solutions.

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

### Using `nix` with flakes  

If you have [Nix](https://nixos.org) installed and flakes enabled, you can set up the dependencies using the provided `flake.nix` and `flake.lock` files.  

#### Set up the environment  

Simply run the following command in the root folder:  

```bash
nix develop
```

This will automatically provide all dependencies specified in the `flake.nix` without needing a virtual environment.  

You're ready to go! 🚀  

## Used Technologies

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-%230C55A5.svg?style=for-the-badge&logo=scipy&logoColor=%white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
