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
sudo pip3 install torchnet==0.0.2
sudo pip3 install torch==0.4.0

```



# Howto

Train WRN-28-10 on 4 GPUs:

```
python3  main.py --save ./logs/resnet_$RANDOM$RANDOM --depth 28 --width 10 --ngpu 2 --gpu_id 0,1 

```

With dropout on my laptop

```
python3  main.py --save ./logs/resnet_dropout/ --depth 28 --width 10 --ngpu 1 --gpu_id 0 --batch_size 32 --dropout_prob 0.2
```
