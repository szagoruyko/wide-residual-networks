#!/usr/bin/env bash

export learningRate=0.01
export epoch_step="{80,120}"
export max_epoch=160
export learningRateDecay=0
export learningRateDecayRatio=0.1
export nesterov=true

export dropout=0.4
export dataset=./datasets/svhn.t7
export randomcrop=0
export hflip=false

# tee redirects stdout both to screen and to file
# have to create folder for script and model beforehand
export save=logs/svhn_${model}_${RANDOM}${RANDOM}
mkdir -p $save
th train.lua | tee $save/log.txt
