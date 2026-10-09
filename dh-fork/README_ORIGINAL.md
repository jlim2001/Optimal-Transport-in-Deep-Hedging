## Code for the paper *Distributional Adversarial Attacks and Training in Deep Hedging*

This repository contains the code to replicate all experimental results of the paper *Distributional Adversarial Attacks and Training in
Deep Hedging*. 

## Requirements

### Installation

Install PyTorch based on your CUDA version:

```bash
# CUDA 11.8
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu
```

Install additional Python package:

```bash
pip install argparse
pip install numpy
pip install panda
```

### Generate Dataset

```sh
python BS_generator.py
python Heston_generator.py
```
Open `GAD_generator.ipynb` and run all cells in the notebook.

All datasets will be stored in `'Data'`


## Training Scripts on BS and Heston model

### Adversarial Training on Black-Scholes Model

```bash
python BS_train_adv.py --N 10000 --delta 0.1 --alpha 1.0
```

### Clean Training on Black-Scholes Model

```bash
python BS_train_clean.py --N 10000
```

### Adversarial Training on Heston Model

```bash
python Heston_train_adv.py --N 10000 --delta 0.1 --alpha 1.0 --attack_method 'S' --transaction_cost_rate 0.0
```

### Clean Training on Heston Model

```bash
python Heston_train_clean.py --N 10000 --transaction_cost_rate 0.0
```

### Input Arguments

* `N`: Number of samples.
* `delta`: Perturbation magnitude of the adversarial attack (applicable only to adversarial training).
* `alpha`: Parameter alpha in Equation (5.1) (applicable only to adversarial training).
* `attack_method`: `'S'` for S-Attack or `'SV'` for SV-Attack (applicable only to adversarial training on the Heston model).
* `transaction_cost_rate`: Transaction cost rate.(applicable only to Heston model).

### Outputs

In each script, we partition the training dataset into smaller subsets with sizes N, so 100000/N neural networks are independently trained on these subsets.
The state dictionaries of the trained networks are saved to the `Result` directory for later evaluation. 

## Training Scripts on real market data

```bash
python GAD_train_adv.py --N 10000 --company 'AAPL' --input 'fix'  --delta 0.1 --alpha 1.0 
```
### Input Arguments
* `N`: Number of samples.
* `company`: Ticker symbol of the company (e.g., `'AAPL'`, `'MSFT'`, `'AMZN'`, `'GOOGL'`, `'BRK-B'`).
* `input`: input dataset (`'fix'` or `'robust'`).
* `delta`: Perturbation magnitude of the adversarial attack. Set `delta=0` for clean training.
* `alpha`: Parameter alpha in Equation (5.1).

### Outputs

Same as above.

## Evaluation

Guidance on assessing network performance on a given dataset are provided in the following notebooks:

* **Black-Scholes Model**: `src/BS_evaluation.ipynb`
* **Heston Model**: `src/Heston_evaluation.ipynb`
* **GAD Model (Real Data)**: `src/GAD_evaluation.ipynb`


## Distributionally Adversarial Attack

In `src/Heston_att.ipynb`, we provide guidance for performing distributionally adversarial attacks as well as comparing autocorrelation functions (ACF) and covariance differences between perturbed and original data as described in Section 5.1 and Appendix E.1.
