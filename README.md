# Low-tubal-rank-tensor-recovery-via-FGD

MATLAB implementation for the paper *"Low-Tubal-Rank Tensor Recovery via Factorized Gradient Descent"* (IEEE TSP 2024).  
Compares Gradient Descent (GD) and Tensor Nuclear Norm (TNN) methods for low-tubal-rank tensor recovery under Gaussian measurements.

## 📝 Table of Contents
- [Overview](#overview)
- [Citation](#citation)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Usage](#usage)
- [License](#license)

## 🌐 Overview
This repository implements two tensor completion approaches:
1. **Gradient Descent (GD)** with over-parameterized initialization
2. **Tensor Nuclear Norm (TNN)** minimization


The experiment compares:
- Convergence speed (iteration vs error)
- Computational efficiency (time vs error)
- Robustness across multiple trials


## 📦 Dependencies
- MATLAB R2018b or newer
- Required Toolboxes: None 
- Third-party Libraries:
  - [tensor-completion-tensor-recovery](https://github.com/your_profile/tensor-completion-tensor-recovery) (include in `/tensor-completion-tensor-recovery-master`)


## 🔧 Setup
1. Clone repository:
   ```bash
   git clone https://github.com/your_profile/tensor-completion-comparison.git
   ```
2. Add dependencies to MATLAB path:
   ```matlab
   addpath(genpath('tensor-completion-tensor-recovery-master'));
   ```


## 🚀 Usage
### Run Main Experiment
Execute `Main_Script.m` to:
- Generate synthetic 3D tensor data (50x50x3)
- Compare GD and TNN methods over 10 trials
- Visualize convergence curves


Key parameters in `Main_Script.m`:
```matlab
n1 = 50;         % Tensor dimension 1
n3 = 3;          % Tubal dimension
tubal_r = 3;     % True tubal rank
k = 3;           % Initial guess rank
ite = 500;       % Max iterations
repeat_time = 10;% Trial repetitions
```

## 📜 License
This project is licensed under the Apache-2.0 License - see the [LICENSE.md](LICENSE.md) file for details.

## ✨ Citation 
If you find this code useful, please cite our work:
```bibtex
@article{liu2024low,
  title={Low-Tubal-Rank Tensor Recovery via Factorized Gradient Descent},
  author={Liu, Zhiyu and Han, Zhi and Tang, Yandong and Zhao, Xi-Le and Wang, Yao},
  journal={IEEE Transactions on Signal Processing},
  year={2024},
  publisher={IEEE}
}
