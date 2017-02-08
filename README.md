Wide Residual Networks
=============

This code was used for experiments with Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.

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
CIFAR-10, CIFAR-100, SVHN, COCO and substantial improvements on ImageNet) and train **several times faster** than pre-activation ResNets.

**Update:** We updated the paper with ImageNet, COCO and meanstd preprocessing CIFAR results.
If you're comparing your method against WRN, please report correct preprocessing numbers because they give substantially different results.

tldr; ImageNet WRN-50-2-bottleneck (ResNet-50 with wider inner bottleneck 3x3 convolution) is significantly faster than ResNet-152 and has better accuracy; on CIFAR meanstd preprocessing (as in fb.resnet.torch) gives better results than ZCA whitening; on COCO wide ResNet with 34 layers outperforms even Inception-v4-based Fast-RCNN model in single model performance.

Test error (%, flip/translation augmentation, **meanstd** normalization, median of 5 runs) on CIFAR:

Network          | CIFAR-10 | CIFAR-100 |
-----------------|:--------:|:--------:
pre-ResNet-164   | 5.46     | 24.33
pre-ResNet-1001  | 4.92     | 22.71
WRN-28-10        | 4.00     | 19.25
WRN-28-10-dropout| **3.89**     | **18.85**

Single-time runs (meanstd normalization):

Dataset | network | test perf. |
--------|:-------:|:---------:|
CIFAR-10  | WRN-40-10-dropout | 3.8%
CIFAR-100 | WRN-40-10-dropout | 18.3%
SVHN      | WRN-16-8-dropout  | 1.54%
ImageNet (single crop) | WRN-50-2-bottleneck | 21.9% top-1, 5.79% top-5
COCO-val5k (single model) | WRN-34-2 | 36 mAP

See http://arxiv.org/abs/1605.07146 for details.

<img src=https://cloud.githubusercontent.com/assets/4953728/20631813/17db5768-b339-11e6-985b-6bbdf4e97aec.png width=440><img src=https://cloud.githubusercontent.com/assets/4953728/20639751/4f50d216-b3cc-11e6-8c24-2c768c917771.png width=440>

bibtex:

```
@INPROCEEDINGS{Zagoruyko2016WRN,
    author = {Sergey Zagoruyko and Nikos Komodakis},
    title = {Wide Residual Networks},
    booktitle = {BMVC},
    year = {2016}}
```

# Pretrained models

## ImageNet

WRN-50-2-bottleneck (wider bottleneck), see [pretrained](pretrained/README.md) for details<br>
Download (263MB): https://yadi.sk/d/-8AWymOPyVZns

There are also PyTorch and Tensorflow model definitions with pretrained weights at
<https://github.com/szagoruyko/functional-zoo/blob/master/wide-resnet-50-2-export.ipynb>

## COCO

Coming

# Installation

The code depends on Torch http://torch.ch. Follow instructions [here](http://torch.ch/docs/getting-started.html) and run:

```
luarocks install torchnet
luarocks install optnet
luarocks install iterm
```

For visualizing training curves we used ipython notebook with pandas and bokeh.

# Usage

## Dataset support

The code supports loading simple datasets in torch format. We provide the following:

* MNIST
[data preparation script](https://gist.github.com/szagoruyko/8467ee15d020ab2a7ce80a215af71b74)
* CIFAR-10
[**recommended**]
[data preparation script](https://gist.github.com/szagoruyko/e5cf5e9b54661a817695c8c7b5c3dfa6),
[preprocessed data (176MB)](https://yadi.sk/d/eFmOduZyxaBrT)
* CIFAR-10 whitened (using pylearn2)
[preprocessed dataset](https://yadi.sk/d/em4b0FMgrnqxy)
* CIFAR-100
[**recommended**]
[data preparation script](https://gist.github.com/szagoruyko/01bfa936396f913a899ee49b98e7304b),
[preprocessed data (176MB)](https://yadi.sk/d/ZbiXAegjxaBcM)
* CIFAR-100 whitened (using pylearn2)
[preprocessed dataset](https://yadi.sk/d/em4b0FMgrnqxy)
* SVHN [data preparation script](https://gist.github.com/szagoruyko/27712564a3f3765c5bfd933b56a21757)

To whiten CIFAR-10 and CIFAR-100 we used the following scripts https://github.com/lisa-lab/pylearn2/blob/master/pylearn2/scripts/datasets/make_cifar10_gcn_whitened.py and then converted to torch using https://gist.github.com/szagoruyko/ad2977e4b8dceb64c68ea07f6abf397b and npy to torch converter https://github.com/htwaijry/npy4th.

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

* NIN (7.4% on CIFAR-10 whitened)
* VGG (modified from [cifar.torch](https://github.com/szagoruyko/cifar.torch), 6.3% on CIFAR-10 whitened)
* pre-activation ResNet (from https://github.com/KaimingHe/resnet-1k-layers)

## Implementation details

The code evolved from https://github.com/szagoruyko/cifar.torch. To reduce memory usage we use @fmassa's [optimize-net](https://github.com/fmassa/optimize-net), which automatically shares output and gradient tensors between modules. This keeps memory usage below 4 Gb even for our best networks. Also, it can generate network graph plots as the one for WRN-16-2 in the end of this page.

# Acknowledgements

We thank startup company [VisionLabs](http://www.visionlabs.ru/en/) and Eugenio Culurciello for giving us access to their clusters, without them ImageNet experiments wouldn't be possible. We also thank Adam Lerer and Sam Gross for helpful discussions. Work supported by EC project FP7-ICT-611145 ROBOSPECT.

<center><img src=https://cloud.githubusercontent.com/assets/4953728/15483030/fc74ec0c-2132-11e6-9e1f-9bc03a83eeea.png width=300></center>
