-- This is a modified version of NIN network in
-- https://github.com/szagoruyko/cifar.torch
require 'nn'

local model = nn.Sequential()

local function Block(...)
  local arg = {...}
  model:add(nn.SpatialConvolution(...))
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
model:add(nn.Linear(192,10))

-- check that we can propagate forward without errors
-- print(#model:float():forward(torch.FloatTensor(1,3,32,32))); model:reset()

for k,v in pairs(model:findModules'nn.SpatialConvolution') do
  v.weight:normal(0,0.05)
  v.bias:zero()
end

return model
