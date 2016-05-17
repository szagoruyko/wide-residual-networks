Code for Wide Residual Networks
=============

This code was used for experiments with Wide Residual Networks http://arxiv.org/abs/1504.03641 by Sergey Zagoruyko and Nikos Komodakis.

The paper shows superiority of wide residual networks (WRN) over thin very deep and that even a simple WRN-16 outperforms thin pre-activation ResNet-1001. We further show that wider and deeper networks achieve much better results, around 1% better than ResNet-1001 on CIFAR-10 and 2.7% better on CIFAR-100. Additionaly, we show state-of-the-art on SVHN using WRN with dropout. WRN trains several times faster than pre-activation ResNets.

# Installation

The code depends on Torch http://torch.ch. After installing it, install additional packages:

```
luarocks install optnet
luarocks install https://raw.githubusercontent.com/szagoruyko/iterm.torch/master/iterm-scm-1.rockspec
```

We highly recommend installing CUDNN-v5 for speed. Optionally you can run on CPU or OpenCL.

# Dataset support

The code supports loading simple datasets in torch format. We provide `t7` files for the follow datasets:

* MNIST
* CIFAR-10
* CIFAR-10 whitened (using pylearn2)
* CIFAR-100
* CIFAR-100 whitened (using pylearn2)
* SVHN

We are currently running ImageNet experiments are will update the paper and this repo as soon as we have them.
The datasets are hosted on yandex disk Download link. Download them and put into datasets folder.

# Running the experiments

We provide several scripts for reproducing results in the paper. Below are several examples.

```bash
CUDA_VISIBLE_DEVICES=0 widen_factor=4 depth=40 dataset=./datasets/cifar10_whitened.t7 ./scripts/train_wrn.sh
```

This will train WRN-40-4 on CIFAR-10 whitened.t7. This network achieves about the same accuracy as ResNet-1001 and trains in X hours on a single Titan X.

```bash
CUDA_VISIBLE_DEVICES=0 widen_factor=10 depth=28 dataset=./datasets/cifar100_whitened.t7 ./scripts/train_wrn.sh
```

This network achieves 20.5% accuracy on CIFAR-100 in about a day on a single Titan X.

For reference we provide logs for this experiment and ipython notebook to visualize the results in notebooks/wrn.ipynb. After running it you should see these training curves:

As WRN are much faster to train than ResNet we don't provide and multi-GPU code. https://github.com/facebook/fb.resnet.torch should be trivial to modify for running WRN.

# Implementation details

The code evolved from https://github.com/szagoruyko/cifar.torch

## Optnet

To reduce memory usage we use @fmassa's optimize-net, which automatically shares output and gradient tensors between modules. This keeps memory usage below 4 Gb even for our widest networks. Also, it can generate network graph plots as the one for WRN-16-2 below.
