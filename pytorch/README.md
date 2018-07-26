PyTorch training code for Wide Residual Networks
==========

PyTorch training code for Wide Residual Networks:
http://arxiv.org/abs/1605.07146

The code reproduces *exactly* it's lua version:
https://github.com/szagoruyko/wide-residual-networks


# Requirements

Install requirements:

```
sudo pip3 install -r requirements.txt 
sudo pip3 install git+https://github.com/pytorch/tnt.git@master

```



# Howto

Train WRN-28-10 on 4 GPUs:

```
python3  main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --ngpu 2 --gpu_id 0,1 
```
