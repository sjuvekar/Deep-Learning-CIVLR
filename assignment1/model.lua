require 'torch'
require 'image'
require 'nn'

-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end
-- end command line

-- Create model
-- 10 class problem
nout = 10

-- Input dimension
nfeats = 3
width = 32
height = 32
nin = nfeats * width * height

print "Creating model...."
model = nn.Sequential()

-- linear
if opt.model == "linear" then
	model:add(nn.Reshape(nin))
	model:add(nn.Linear(nin, nout))
-- mlp
elseif opt.model == 'mpl' then
	nhidden = nin / 2
	model:add(nn.Reshape(nin))
	model:add(nn.Linear(nin, nhidden))
	model:add(nn.Tanh())
	model:add(nn.Linear(nhidden, nout))
elseif opt.model == 'convnet' then

  -- hidden units, filter sizes (for ConvNet only):
  nstates = {64,64,128}
  filtsize = 5
  poolsize = 2
  normkernel = image.gaussian1D(7)

  if opt.type == 'cuda' then
    -- Special modern convnet for cuda
    -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

    -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

    -- stage 3 : standard 2-layer neural network
    model:add(nn.View(nstates[2]*filtsize*filtsize))
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
    model:add(nn.ReLU())
    model:add(nn.Linear(nstates[3], nout))

	else
  	-- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
    model:add(nn.Tanh())
    model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
    model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

    -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
    model:add(nn.Tanh())
    model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
    model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

    -- stage 3 : standard 2-layer neural network
    model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
    model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
    model:add(nn.Tanh())
    model:add(nn.Linear(nstates[3], nout))
  end
  
end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)
----------------------------------------------------------------------

-- Visualization is quite easy, using itorch.image().

if itorch then
	if opt.model == 'convnet' then
		print '==> visualizing ConvNet filters'
		print('Layer 1 filters:')
		itorch.image(model:get(1).weight)
		print('Layer 2 filters:')
		itorch.image(model:get(5).weight)
	end
end
