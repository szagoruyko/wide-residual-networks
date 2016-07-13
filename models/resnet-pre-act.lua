--  ResNet-1001
--  This is a re-implementation of the 1001-layer residual networks described in:
--  [a] "Identity Mappings in Deep Residual Networks", arXiv:1603.05027, 2016,
--  authored by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

--  Acknowledgement: This code is contributed by Xiang Ming from Xi'an Jiaotong Univeristy.

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
local utils = paths.dofile'utils.lua'

local Convolution = nn.SpatialConvolution
local Avg = nn.SpatialAveragePooling
local ReLU = nn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   
   -- The new Residual Unit in [a]
   local function bottleneck(nInputPlane, nOutputPlane, stride)
      
      local nBottleneckPlane = nOutputPlane / 4
      if opt.resnet_nobottleneck then
         nBottleneckPlane = nOutputPlane
      end
      
      if nInputPlane == nOutputPlane then -- most Residual Units have this shape      
         local convs = nn.Sequential()
         -- conv1x1
         convs:add(SBatchNorm(nInputPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
        
         -- conv3x3
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
        
         -- conv1x1
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))
        
         local shortcut = nn.Identity()
        
         return nn.Sequential()
            :add(nn.ConcatTable()
               :add(convs)
               :add(shortcut))
            :add(nn.CAddTable(true))
      else -- Residual Units for increasing dimensions
         local block = nn.Sequential()
         -- common BN, ReLU
         block:add(SBatchNorm(nInputPlane))
         block:add(ReLU(true))
        
         local convs = nn.Sequential()     
         -- conv1x1
         convs:add(Convolution(nInputPlane,nBottleneckPlane,1,1,stride,stride,0,0))
        
         -- conv3x3
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,3,3,1,1,1,1))
        
         -- conv1x1
         convs:add(SBatchNorm(nBottleneckPlane))
         convs:add(ReLU(true))
         convs:add(Convolution(nBottleneckPlane,nOutputPlane,1,1,1,1,0,0))
        
         local shortcut = nn.Sequential()
         shortcut:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
        
         return block
            :add(nn.ConcatTable()
               :add(convs)
               :add(shortcut))
            :add(nn.CAddTable(true))
      end
   end

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
      assert((depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001 in the paper)')
      local n = (depth - 2) / 9

      -- The new ResNet-164 and ResNet-1001 in [a]
      local nStages = {16, 64, 128, 256}

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

   utils.DisableBias(model)
   utils.testModel(model)
   utils.MSRinit(model)
   utils.FCinit(model)

   -- model:get(1).gradInput = nil

   return model
end

return createModel
