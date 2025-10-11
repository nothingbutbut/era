# ResNet50-ERA
This repository contains the code for the paper **"Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints"**([Link to Paper](https://arxiv.org/abs/2510.08549)). Built on top of the ResNet50 architecture, it introduces the ERA (Entropy Regularized Activation) function to enhance performance in image classification tasks.

This codebase is adapted from [timm](https://github.com/rwightman/pytorch-image-models), and we sincerely thank the author Ross Wightman for his excellent work.
We keep most of the original code and only modify the output activation of the resnet model.

**Notice**: ERA is only implemented and tested in the ResNet50 architecture currently. We will extend it to other architectures in the future.

## 📰Updates
- **2025.10.06**: We build the ResNet50-ERA codebase based on the original timm repository.

## 🎯Features
- Implementation of the ERA (Entropy Regularized Activation) function.
- Adaptation of the ResNet50 architecture to incorporate ERA.
- Enhanced performance in image classification tasks.

## 📖Installation
Simply run:
```bash
pip install -e .
```

## 🚀Usage
For a quick start, you can run the following command to train a ResNet50-ERA model on the CIFAR-10 dataset:
```bash
./distributed_train.sh 4 --data-dir ../data --dataset torch/cifar10 --dataset-download -b 64 --model resnet50 --sched cosine --epochs 200 --lr 0.05 --amp --remode pixel --reprob 0.6 --aug-splits 3 --aa rand-m9-mstd0.5-inc1 --resplit --split-bn --jsd --dist-bn reduce
```
Make sure to adjust the parameters according to your setup and requirements. For more detailed options and configurations, refer to the timm documentation.