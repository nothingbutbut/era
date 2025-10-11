# 💥 SAC-ERA
This repository contains the code for the paper **"Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints"**([Link to Paper](https://arxiv.org/abs/2510.08549)). Built on top of the SAC (Soft Actor-Critic) framework, it introduces the ERA (Entropy Regularized Activation) function to enhance performance in various tasks.

## 📰Updates
- **2025.09.21**: Repository created and initial code for SAC with ERA activation function added.

## 🎯Features
- Implementation of the ERA activation function.
- Integration with the SAC algorithm for continuous control tasks.
- Support for training on various environments, including DeepMind Control Suite and HumanoidBench.
- Parallel training support for multiple seeds.
- Use Wandb for experiment tracking.

## 📖Installation
First, Install prerequisites:
```bash
sudo apt install libglfw3 libgl1-mesa-glx libosmesa6-dev patchelf
```
Then create the conda environment:
```bash
conda env create -f jaxrl.yml
conda activate jaxrl
```
**Optional**: If you want to do HumanoidBench tasks, still need to install HumanoidBench:
```bash
pip install --editable git+https://github.com/carlosferrazza/humanoid-bench.git#egg=humanoid-bench
```
Then install JAX with CUDA support (for CUDA 12.0 and cuDNN 8.9):
```bash
pip install -U jaxlib==0.4.19+cuda12.cudnn89 jax==0.4.19 jaxopt==0.8 numpy==1.25.2 scipy==1.11.2 ml-dtypes==0.3.1 opt-einsum==3.3.0  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --force-reinstall
```

**If HumanoidBench is installed**: Additionally, if you have installed HumanoidBench, you may need to re-install `torch` as humanoidbench's dependency may conflict with JAX.
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```
Then run the following command to install nvidia related packages that are compatible with both JAX and PyTorch:
```bash
pip install -r hb_reqs.txt
```

## 🚀Usage
For debug run:
```bash
conda activate jaxrl
# For DeepMind Control Suite (DMC) tasks:
python train_parallel.py --benchmark dmc --env_name dog-trot --debug
# For HumanoidBench tasks:
python train_parallel.py --benchmark hb --env_name h1-walk-v0 --debug
```

To train a SAC agent with ERA activation on a specific environment, use the following command:
```bash
conda activate jaxrl
# For DeepMind Control Suite (DMC) tasks:
python train_parallel.py --benchmark dmc --env_name dog-trot --num_seeds 3
# For HumanoidBench tasks:
python train_parallel.py --benchmark hb --env_name h1-walk-v0 --num_seeds 3
```