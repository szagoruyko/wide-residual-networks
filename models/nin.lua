-- This is a modified version of NIN network in
-- https://github.com/szagoruyko/cifar.torch
-- Network-In-Network: http://arxiv.org/abs/1312.4400
-- Modifications:
--  * removed dropout
--  * added BatchNorm
--  * the last layer changed from avg-pooling to linear (works better)
require 'nn'
local utils = paths.dofile'utils.lua'

local function createModel(opt)
   local model = nn.Sequential()

   local function Block(...)
     local arg = {...}
     model:add(nn.SpatialConvolution(...):noBias())
     model:add(nn.SpatialBatchNormalization(arg[2],1e-5))
     model:add(nn.ReLU(true))
     return model
   end

   Block(3,192,5,5,1,1,2,2)
   Block(192,160,1,1)
   Block(160,96,1,1)
   model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
   Block(96,192,5,5,1,1,2,2)
   Block(192,192,1,1)
   Block(192,192,1,1)
   model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
   Block(192,192,3,3,1,1,1,1)
   Block(192,192,1,1)
   Block(192,192,1,1)
   model:add(nn.SpatialAveragePooling(8,8,1,1))
   model:add(nn.View(-1):setNumInputDims(3))
   model:add(nn.Linear(192,opt and opt.num_classes or 10))

   utils.FCinit(model)
   utils.testModel(model)
   utils.MSRinit(model)
   return model
end

return createModel
