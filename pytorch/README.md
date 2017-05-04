PyTorch training code for Wide Residual Networks
==========

PyTorch training code for Wide Residual Networks:
http://arxiv.org/abs/1605.07146

The code reproduces *exactly* it's lua version:
https://github.com/szagoruyko/wide-residual-networks


# Requirements

The script depends on opencv python bindings, easily installable via conda:

```
conda install -c menpo opencv3
```

After that and after installing pytorch do:

```
pip install -r requirements.txt
```


# Howto

Train WRN-28-10 on 4 GPUs:

```
python main.py main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --ngpu 4 --gpu_id 0,1,2,3
```
