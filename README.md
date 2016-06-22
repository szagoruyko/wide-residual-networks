Code for Wide Residual Networks
=============

This code was used for experiments with Wide Residual Networks http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.

Deep residual networks were shown to be able to scale up to thousands of
layers and still have improving performance. However, each fraction of a
percent of improved accuracy costs nearly doubling the number of layers, and so
training very deep residual networks has a problem of diminishing feature
reuse, which makes these networks very slow to train. 

To tackle these problems,
in this work we conduct a detailed experimental study on the architecture of
ResNet blocks, based on which we propose a novel architecture where we *decrease
depth* and *increase width* of residual networks. We call the resulting network
structures **wide residual networks (WRNs)** and show that these are far superior
over their commonly used thin and very deep counterparts.

For example, we
demonstrate that even a simple 16-layer-deep wide residual network outperforms
in accuracy and efficiency all previous deep residual networks, including
thousand-layer-deep networks. We further show that WRNs achieve **incredibly** 
good results (e.g., achieving new state-of-the-art results on
CIFAR-10, CIFAR-100 and SVHN) and train **several times faster** than pre-activation ResNets.

Test error (%, flip/translation augmentation) on CIFAR:

Method | CIFAR-10 | CIFAR-100
-------|:--------:|:--------:
pre-ResNet-164 | 5.46 | 24.33
pre-ResNet-1001 | 4.92 | 22.71
WRN-28-10 | **4.17** | 20.5
WRN-28-10-dropout| 4.39 | **20.0**

See http://arxiv.org/abs/1605.07146 for details.

<img src=https://cloud.githubusercontent.com/assets/4953728/15482554/91f041da-2130-11e6-87be-d3cee0867ac5.png width=440><img src=https://cloud.githubusercontent.com/assets/4953728/15482555/9217de66-2130-11e6-9a25-8d0ff4f07e15.png width=440>

# Installation

The code depends on Torch http://torch.ch. Follow instructions [here](http://torch.ch/docs/getting-started.html) and run:

```
luarocks install optnet
luarocks install iterm
```

We recommend installing CUDNN v5 for speed. Alternatively you can run on CPU or on GPU with OpenCL (coming).

For visualizing training curves we used ipython notebook with pandas and bokeh and suggest using anaconda.

# Usage

## Dataset support

The code supports loading simple datasets in torch format. We provide the following:

* MNIST [data preparation script](https://gist.github.com/szagoruyko/8467ee15d020ab2a7ce80a215af71b74)
* CIFAR-10 (coming)
* CIFAR-10 whitened (using pylearn2) [preprocessed dataset](https://yadi.sk/d/em4b0FMgrnqxy)
* CIFAR-100 (coming)
* CIFAR-100 whitened (using pylearn2) [preprocessed dataset](https://yadi.sk/d/em4b0FMgrnqxy)
* SVHN [data preparation script](https://gist.github.com/szagoruyko/27712564a3f3765c5bfd933b56a21757)

To whiten CIFAR-10 and CIFAR-100 we used the following scripts https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py and then converted to torch using https://gist.github.com/szagoruyko/ad2977e4b8dceb64c68ea07f6abf397b.

We are running ImageNet experiments and will update the paper and this repo soon.

## Training

We provide several scripts for reproducing results in the paper. Below are several examples.

```bash
model=wide-resnet widen_factor=4 depth=40 ./scripts/train_cifar.sh
```

This will train WRN-40-4 on CIFAR-10 whitened (supposed to be in `datasets` folder). This network achieves about the same accuracy as ResNet-1001 and trains in 6 hours on a single Titan X. 
Log is saved to `logs/wide-resnet_$RANDOM$RANDOM` folder with json entries for each epoch and can be visualized with itorch/ipython later.

For reference we provide logs for this experiment and [ipython notebook](notebooks/visualize.ipynb) to visualize the results. After running it you should see these training curves:

![viz](https://cloud.githubusercontent.com/assets/4953728/15482840/11b46698-2132-11e6-931e-04680ae42c3c.png)

Another example:

```bash
model=wide-resnet widen_factor=10 depth=28 dropout=0.3 dataset=./datasets/cifar100_whitened.t7 ./scripts/train_cifar.sh
```

This network achieves 20.0% error on CIFAR-100 in about a day on a single Titan X.

Multi-GPU is supported with `nGPU=n` parameter.

## Other models

Additional models in this repo:

* NIN (7.4% on CIFAR-10)
* VGG (modified from [cifar.torch](https://github.com/szagoruyko/cifar.torch), 6.3% on CIFAR-10)
* pre-activation ResNet (from https://github.com/KaimingHe/resnet-1k-layers)

## Implementation details

The code evolved from https://github.com/szagoruyko/cifar.torch. To reduce memory usage we use @fmassa's [optimize-net](https://github.com/fmassa/optimize-net), which automatically shares output and gradient tensors between modules. This keeps memory usage below 4 Gb even for our best networks. Also, it can generate network graph plots as the one for WRN-16-2 below.

<center><img src=https://cloud.githubusercontent.com/assets/4953728/15483030/fc74ec0c-2132-11e6-9e1f-9bc03a83eeea.png width=300></center>
