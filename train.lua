-- Code for Wide Residual Networks http://arxiv.org/abs/1605.07146
-- (c) Sergey Zagoruyko, 2016
require 'xlua'
require 'optim'
require 'image'
local tnt = require 'torchnet'
local c = require 'trepl.colorize'
local json = require 'cjson'
local utils = paths.dofile'models/utils.lua'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'

local opt = {
  dataset = './datasets/cifar10_whitened.t7',
  save = 'logs',
  batchSize = 128,
  learningRate = 0.1,
  learningRateDecay = 0,
  learningRateDecayRatio = 0.2,
  weightDecay = 0.0005,
  dampening = 0,
  momentum = 0.9,
  epoch_step = "80",
  max_epoch = 300,
  model = 'nin',
  optimMethod = 'sgd',
  init_value = 10,
  depth = 50,
  shortcutType = 'A',
  nesterov = false,
  dropout = 0,
  hflip = true,
  randomcrop = 4,
  imageSize = 32,
  randomcrop_type = 'zero',
  cudnn_deterministic = false,
  optnet_optimize = true,
  generate_graph = false,
  multiply_input_factor = 1,
  widen_factor = 1,
  nGPU = 1,
  data_type = 'torch.CudaTensor',
  seed = 444,
}
opt = xlua.envparams(opt)


torch.manualSeed(opt.seed)

opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
print(opt)

local meanstd = {mean = {125.3, 123.0, 113.9}, std  = {63.0,  62.1,  66.7}}
print(c.blue '==>' ..' loading data')
local provider = torch.load(opt.dataset)
opt.num_classes = provider.testData.labels:max()
if torch.type(provider.trainData.data) == 'torch.ByteTensor' then
   for i,v in ipairs{'trainData', 'testData'} do
      provider[v].data = provider[v].data:float()--:div(256)
      for ch=1,3 do
         provider[v].data:select(2,ch):add(-meanstd.mean[ch]):div(meanstd.std[ch])
      end
   end
end

local function cast(x) return x:type(opt.data_type) end

print(c.blue '==>' ..' configuring model')
local model = nn.Sequential()
local net = dofile('models/'..opt.model..'.lua')(opt)
if opt.data_type:match'torch.Cuda.*Tensor' then
   require 'cudnn'
   require 'cunn'
   cudnn.convert(net, cudnn):cuda()
   if opt.cudnn_deterministic then
      net:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
   end

   print(net)
   print('Network has', #net:findModules'cudnn.SpatialConvolution', 'convolutions')

   local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
   if opt.generate_graph then
      iterm.dot(graphgen(net, sample_input), opt.save..'/graph.pdf')
   end
   if opt.optnet_optimize then
      optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
   end
   -- to avoid optnet messing cudnn FindEx
   cudnn.benchmark = true
end
model:add(utils.makeDataParallelTable(net, opt.nGPU))
cast(model)

local function hflip(x)
   return torch.random(0,1) == 1 and x or image.hflip(x)
end

local function randomcrop(x)
   local pad = opt.randomcrop
   if opt.randomcrop_type == 'reflection' then
      module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float()
   elseif opt.randomcrop_type == 'zero' then
      module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
   else
      error'unknown mode'
   end

   local imsize = opt.imageSize
   local padded = module:forward(x)
   local x = torch.random(1,pad*2 + 1)
   local y = torch.random(1,pad*2 + 1)
   return padded:narrow(3,x,imsize):narrow(2,y,imsize)
end


local function getIterator(mode)
   local dataset = provider[mode..'Data']

   local list_dataset = tnt.ListDataset{
      list = torch.range(1, dataset.labels:numel()):long(),
      load = function(idx)
         return {
            input = dataset.data[idx]:float(),
            target = torch.LongTensor{dataset.labels[idx]},
         }
      end,
   }

   local d = mode == 'train' and list_dataset
         :shuffle()
         :transform{
            input = tnt.transform.compose{
               opt.hflip and hflip or nil,
               opt.randomcrop > 0 and randomcrop or nil,
            }
         }
         :batch(opt.batchSize, 'skip-last')
      or list_dataset
         :batch(opt.batchSize, 'include-last')

   function d:manualSeed(seed) torch.manualSeed(seed) end

   return tnt.ParallelDatasetIterator{
      nthread = 2,
      init = function()
         require 'torchnet'
         require 'image'
         require 'nn'
      end,
      closure = function()
         return d
      end,
   }
end

local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

print('Will save at '..opt.save)
paths.mkdir(opt.save)

local engine = tnt.OptimEngine()
local criterion = cast(nn.CrossEntropyCriterion())
local meter = tnt.AverageValueMeter()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local train_timer = torch.Timer()
local test_timer = torch.Timer()

engine.hooks.onStartEpoch = function(state)
   local epoch = state.epoch + 1
   print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   meter:reset()
   clerr:reset()
   train_timer:reset()
   if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 or
      torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
      opt.learningRate = opt.learningRate * opt.learningRateDecayRatio
      state.config = tablex.deepcopy(opt)
      state.optim = tablex.deepcopy(opt)
   end

   state.iterator:exec('manualSeed', state.epoch + 1)
   state.iterator:exec'resample'
end

engine.hooks.onEndEpoch = function(state)
   local train_loss = meter:value()
   local train_err = clerr:value{k = 1}
   local train_time = train_timer:time().real
   meter:reset()
   clerr:reset()
   test_timer:reset()

   engine:test{
      network = model,
      iterator = getIterator('test'),
      criterion = criterion,
   }

   log{
      loss = train_loss,
      train_loss = train_loss,
      train_acc = 100 - train_err,
      epoch = state.epoch,
      test_acc = 100 - clerr:value{k = 1},
      lr = opt.learningRate,
      train_time = train_time,
      test_time = test_timer:time().real,
      n_parameters = state.params:numel(),
   }
end

engine.hooks.onForwardCriterion = function(state)
   meter:add(state.criterion.output)
   clerr:add(state.network.output, state.sample.target)
end

local inputs = cast(torch.Tensor())
local targets = cast(torch.Tensor())
engine.hooks.onSample = function(state)
   inputs:resize(state.sample.input:size()):copy(state.sample.input)
   targets:resize(state.sample.target:size()):copy(state.sample.target)
   state.sample.input = inputs
   state.sample.target = targets
end

engine:train{
   network = model,
   iterator = getIterator('train'),
   criterion = criterion,
   optimMethod = optim.sgd,
   config = tablex.deepcopy(opt),
   maxepoch = opt.max_epoch,
}

torch.save(opt.save..'/model.t7', net:clearState())
