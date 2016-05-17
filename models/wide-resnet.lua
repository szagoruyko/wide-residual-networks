--  Wide Residual Network
--  This is an implementation of the wide residual networks described in:
--  [a] "Wide Residual Networks", arXiv:1603.05027, 2016,
--  authored by Sergey Zagoruyko and Nikos Komodakis

--  ************************************************************************
--  This code incorporates material from:

--  fb.resnet.torch (https://github.com/facebook/fb.resnet.torch)
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ************************************************************************

local nn = require 'nn'
require 'cunn'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function Dropout()
   return nn[opt.dropout_type](opt.dropout and opt.dropout or 0,nil,true)
end

local function createModel(opt)
   local depth = opt.depth

   local blocks = {}
   
   local function bottleneck(conv_params, nInputPlane, nOutputPlane, stride)
      local nBottleneckPlane = nOutputPlane / 4
      if opt.resnet_nobottleneck then
         nBottleneckPlane = nOutputPlane
      end

      local block = nn.Sequential()
      local convs = nn.Sequential()     

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(SBatchNorm(nInputPlane)):add(ReLU(true))
            local u = v
            u[3], u[4] = stride, stride
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(u)))
         else
            for j=1,opt.resnet_pre_act_inner do
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            if opt.dropout > 0 then
               convs:add(Dropout())
            end
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
            end
         end
      end
     
      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)
     
      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   function blocks.bottleneck(...)
      return bottleneck({
         {1,1,-1,-1,0,0},
         {3,3,1,1,1,1},
         {1,1,1,1,0,0},
      },...)
   end

   local bottleneck = blocks[opt.blocktype]
   assert(bottleneck)

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()

      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   local model = nn.Sequential()
   do
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6
      print(' | ResNet-' .. depth .. ' CIFAR-10')

      local nStages = torch.Tensor{16, 16, 32, 64}
      nStages:narrow(1,2,nStages:size(1)-1):mul(opt.widen_factor)

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(layer(bottleneck, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(bottleneck, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(bottleneck, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(SBatchNorm(nStages[4]))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      model:add(nn.Linear(nStages[4], opt.num_classes))
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('nn.SpatialConvolution')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   -- model:get(1).gradInput = nil

   return model
end

return createModel(opt)
