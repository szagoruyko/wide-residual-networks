local utils = {}

function utils.MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

function utils.FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

function utils.DisableBias(model)
   for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
      v.bias = nil
      v.gradBias = nil
   end
end

function utils.testModel(model)
   model:float()
   local imageSize = opt and opt.imageSize or 32
   local input = torch.randn(1,3,imageSize,imageSize):type(model._type)
   print('forward output',{model:forward(input)})
   print('backward output',{model:backward(input,model.output)})
   model:reset()
end

return utils
