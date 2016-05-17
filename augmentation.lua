require 'image'
require 'nn'

do -- random crop
   local RandomCrop, parent = torch.class('nn.RandomCrop', 'nn.Module')

   function RandomCrop:__init(pad, mode)
      assert(pad)
      parent.__init(self)
      self.pad = pad
      if mode == 'reflection' then
         self.module = nn.SpatialReflectionPadding(pad,pad,pad,pad)
      elseif mode == 'zero' then
         self.module = nn.SpatialZeroPadding(pad,pad,pad,pad)
      else
         error'unknown mode'
      end
      self.train = true
   end

   function RandomCrop:updateOutput(input)
      assert(input:dim() == 4)
      local imsize = input:size(4)
      if self.train then
         local padded = self.module:forward(input)
         local x = torch.random(1,self.pad*2 + 1)
         local y = torch.random(1,self.pad*2 + 1)
         self.output = padded:narrow(4,x,imsize):narrow(3,y,imsize)
      else
         self.output:set(input)
      end
      return self.output
   end

   function RandomCrop:type(type)
      self.module:type(type)
      return parent.type(self, type)
   end
end

do -- random horizontal flip
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:updateOutput(input)
    self.train = self.train == nil and true or self.train
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:resize(input:size()):copy(input)
    return self.output
  end
end

