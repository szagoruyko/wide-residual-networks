-- This is a modified version of VGG network in
-- https://github.com/szagoruyko/cifar.torch
-- Modifications:
--  * removed dropout
--  * last nn.Linear layers substituted with convolutional layers
--    and avg-pooling
require 'nn'
local utils = paths.dofile'utils.lua'

local function createModel(opt)
   local model = nn.Sequential()

   -- building block
   local function Block(nInputPlane, nOutputPlane)
      model:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1):noBias())
      model:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
      model:add(nn.ReLU(true))
      return model
   end

   local function MP()
      model:add(nn.SpatialMaxPooling(2,2,2,2):ceil())
      return model
   end

   local function Group(ni, no, N, f)
      for i=1,N do
         Block(i == 1 and ni or no, no)
      end
      if f then f() end
   end

   Group(3,64,2,MP)
   Group(64,128,2,MP)
   Group(128,256,4,MP)
   Group(256,512,4,MP)
   Group(512,512,4)
   model:add(nn.SpatialAveragePooling(2,2,2,2):ceil())
   model:add(nn.View(-1):setNumInputDims(3))
   model:add(nn.Linear(512,opt and opt.num_classes or 10))

   utils.FCinit(model)
   utils.testModel(model)
   utils.MSRinit(model)

   return model
end

return createModel
