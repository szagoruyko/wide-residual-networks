require 'xlua'
require 'optim'
require 'image'
require 'cunn'
require 'cudnn'
local c = require 'trepl.colorize'
local json = require 'cjson'
paths.dofile'augmentation.lua'

-- for memory optimizations and graph generation
local optnet = require 'optnet'
local graphgen = require 'optnet.graphgen'
local iterm = require 'iterm'
require 'iterm.dot'

opt = {
  dataset = './datasets/cifar10_whitened.t7',
  save = 'logs',
  batchSize = 128,
  learningRate = 1,
  learningRateDecay = 1e-7,
  learningRateDecayRatio = 0.5,
  weightDecay = 0.0005,
  dampening = 0,
  momentum = 0.9,
  epoch_step = "25",
  max_epoch = 300,
  model = 'alexnet',
  optimMethod = 'sgd',
  init_value = 10,
  depth = 50,
  shortcutType = 'B',
  nesterov = false,
  dropout = 0,
  hflip = true,
  randomcrop = 4,
  imageSize = 32,
  randomcrop_type = 'zero',
  cudnn_fastest = true,
  optnet_optimize = false,
  generate_graph = false,
  multiply_input_factor = 1,
  widen_factor = 1,
}
opt = xlua.envparams(opt)

opt.epoch_step = tonumber(opt.epoch_step) or loadstring('return '..opt.epoch_step)()
print(opt)

print(c.blue '==>' ..' loading data')
local provider = torch.load(opt.dataset)
opt.num_classes = provider.testData.labels:max()

print(c.blue '==>' ..' configuring model')
local net = dofile('models/'..opt.model..'.lua'):cuda()
local model = nn.Sequential()
local function add(flag, module) if flag then model:add(module) end end
add(opt.hflip, nn.BatchFlip():float())
add(opt.randomcrop > 0, nn.RandomCrop(opt.randomcrop, opt.randomcrop_type):float())
model:add(nn.Copy('torch.FloatTensor','torch.CudaTensor'):cuda())
model:add(net)

cudnn.convert(net, cudnn)
cudnn.benchmark = true
if opt.cudnn_fastest then
   for i,v in ipairs(net:findModules'cudnn.SpatialConvolution') do v:fastest() end
end
if opt.cudnn == 'deterministic' then
   model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
end

print(net)
print('Network has', #model:findModules'cudnn.SpatialConvolution', 'convolutions')

local sample_input = torch.randn(8,3,opt.imageSize,opt.imageSize):cuda()
if opt.generate_graph then
   iterm.dot(graphgen(net, sample_input), opt.save..'/graph.pdf')
end
if opt.optnet_optimize then
   optnet.optimizeMemory(net, sample_input, {inplace = false, mode = 'training'})
end

local function log(t) print('json_stats: '..json.encode(tablex.merge(t,opt,true))) end

print('Will save at '..opt.save)
paths.mkdir(opt.save)

local parameters,gradParameters = model:getParameters()

opt.n_parameters = parameters:numel()
print('Network has ', parameters:numel(), 'parameters')

print(c.blue'==>' ..' setting criterion')
criterion = nn.CrossEntropyCriterion():cuda()

-- in case we want autograd
local f = function(inputs, targets)
   model:forward(inputs)
   local loss = criterion:forward(model.output, targets)
   local df_do = criterion:backward(model.output, targets)
   model:backward(inputs, df_do)
   return loss
end

print(c.blue'==>' ..' configuring optimizer')
optimState = tablex.deepcopy(opt)
optimMethod = optim[opt.optimMethod]


function train()
  model:training()

  local loss = 0

  local targets = torch.CudaTensor(opt.batchSize)
  local indices = torch.randperm(provider.trainData.data:size(1)):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  for t,v in ipairs(indices) do
    local inputs = provider.trainData.data:index(1,v)
    targets:copy(provider.trainData.labels:index(1,v))

    local feval = function(x)
      if x ~= parameters then parameters:copy(x) end
      gradParameters:zero()
      loss = loss + f(inputs, targets)
      return f,gradParameters
    end
    optimMethod(feval, parameters, optimState)
  end

  return loss
end


function test()
  local confusion = optim.ConfusionMatrix(opt.num_classes)

  model:evaluate()
  local bs = opt.batchSize
  local data_split = provider.testData.data:split(bs,1)
  local labels_split = provider.testData.labels:split(bs,1)
  for i,v in ipairs(data_split) do
    local outputs = model:forward(v)
    confusion:batchAdd(outputs, labels_split[i])
  end

  confusion:updateValids()
  local test_acc = confusion.totalValid * 100
  return test_acc
end


for epoch=1,opt.max_epoch do
  print('==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
  local function updateLR(lr)
    opt.learningRate = lr or opt.learningRate * opt.learningRateDecayRatio
    optimState = tablex.deepcopy(opt)
  end
  optimState.learningRate = opt.learningRate / opt.learningRateDecayRatio
  if torch.type(opt.epoch_step) == 'number' and epoch % opt.epoch_step == 0 then
     updateLR()
  elseif torch.type(opt.epoch_step) == 'table' and tablex.find(opt.epoch_step, epoch) then
     updateLR()
  end

  local function t(f) local s = torch.Timer(); return f(), s:time().real end

  local loss, train_time = t(train)
  local test_acc, test_time = t(test)

  log{
     loss = loss,
     epoch = epoch,
     test_acc = test_acc,
     lr = opt.learningRate,
     train_time = train_time,
     test_time = test_time,
   }
end

torch.save(opt.save..'/model.t7', net:clearState())
