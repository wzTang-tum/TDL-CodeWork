# Robust-DPFL: Robust and Differentially Private Federated Learning

This repository implements a simulation framework for **Robust and Differentially Private Federated Learning (Robust-DPFL)**. It investigates the performance of Federated Learning (FL) systems under the dual challenges of **data poisoning attacks** and **privacy constraints** (using Differential Privacy).

> **Note:** This project serves as a research report and study based on the paper **"Towards the Robustness of Differentially Private Federated Learning"**. The codebase is built upon and improves the open-source implementation provided by the original authors.

The project simulates a scenario where a fraction of clients are malicious (performing backdoor attacks) while the central server or clients apply defense mechanisms and differential privacy noise to protect user data.

## 📂 Code Structure

The core logic is located in the `Code/` directory:

- **`Code/Main.py`**: The main entry point of the application. It parses command-line arguments, assigns devices (GPU/CPU), loads data, and orchestrates the FL simulation.
- **`Code/FLTrain.py`**: Contains the core Federated Learning loop. It implements:
    - Local model training (Client side).
    - **Defense Mechanisms**: `RobustDPFL` (Proposed), `FedAvg`, `Krum`, `Mkrum`, `Median` (Mid), `Norm-bounding`.
    - **Differential Privacy**: Implements Local Differential Privacy (LDP) mechanisms via noise addition.
    - **Aggregation Logic**: How the server aggregates weights from multiple clients.
- **`Code/model.py`**: Defines the Convolutional Neural Network (CNN) architectures used for MNIST, FEMNIST, and CIFAR10 datasets.
- **`Code/poison.py`**: Implements **Data Poisoning** and **Backdoor Attacks**. It handles the generation of triggers and their insertion into the training data of malicious clients.
- **`Code/preprocessing.py`**: Handles data loading and partitioning.
    - Supports **IID** and **Non-IID** data distribution (using Dirichlet distribution).
    - Loaders for MNIST, FEMNIST, and CIFAR10.

## 🛠️ Requirements

The project relies on the following Python libraries:
- `tensorflow` (Code runs in TF1 compatibility mode)
- `numpy`
- `scikit-learn`
- `click` (for CLI argument parsing)
- `scipy`

Install dependencies using:
```bash
pip install -r Code/requirement.txt
```

## 🚀 Usage

The simulation is controlled via `Code/Main.py`. You should run the script from the `Code/` directory to ensure relative paths to `Data/` are resolved correctly.

### Basic Command
```bash
cd Code
python Main.py
```

### Command Line Arguments

You can customize the simulation using the following flags:

| Argument | Flag | Default | Description |
|----------|------|---------|-------------|
| **Dataset** | `-d`, `--dataset` | `MNIST` | Target dataset: `MNIST`, `FEMNIST`, or `CIFAR10`. |
| **Attack Mode** | `-a`, `--attack-mode` | `AttackDPFL` | The attack strategy employed by malicious clients. |
| **Defense Mode** | `-m`, `--defense-mode` | `RobustDPFL` | Aggregation/Defense method: `RobustDPFL`, `FedAvg`, `Krum`, `Mkrum`, `Mid`, `Norm`, `Contra`. |
| **Malicious Ratio** | `-t`, `--taxic-ratio` | `0.1` | Fraction of clients that are malicious (e.g., `0.1` = 10%). |
| **Privacy Alpha** | `-o`, `--alpha` | `1.2` | Parameter related to privacy accounting (Rényi DP / Alpha). |
| **Data Distribution** | `--dirichlet-alpha` | `0.5` | Controls the Non-IID degree of data distribution. Smaller values = more Non-IID. |
| **Privacy Budget** | `-e`, `--epsilon` | `5` | Privacy budget epsilon ($\epsilon$). Smaller values = higher privacy (more noise). |
| **GPU** | `-g`, `--gpu` | `0` | GPU device index to use. |

### Examples

**1. Standard FedAvg with no specific defense (Baseline):**
```bash
python Main.py -d MNIST -m FedAvg
```

**2. Run with the proposed Robust-DPFL defense on CIFAR10:**
```bash
python Main.py -d CIFAR10 -m RobustDPFL -e 2.0
```

**3. Test Robustness against 20% malicious clients:**
```bash
python Main.py -d FEMNIST -m RobustDPFL -t 0.2
```

**4. Experiment with Non-IID data distributions (Dirichlet alpha):**
```bash
# High Non-IID (alpha=0.1)
python Main.py -d MNIST --dirichlet-alpha 0.1

# Low Non-IID (alpha=10.0)
python Main.py -d MNIST --dirichlet-alpha 10.0
```

## 📊 Data Partitioning & Attack Implementation

- **Data Partitioning**: The simulation uses a Dirichlet distribution to create realistic Non-IID data splits among 100 clients.
- **Attacks**: Malicious clients inject a specific trigger pattern (defined in `poison.py`) into their local datasets and label them as a target class, attempting to install a backdoor in the global model.
