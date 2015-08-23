require 'torch'
require 'nn'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Loss Function')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
   cmd:text()
   opt = cmd:parse(arg or {})

   -- to enable self-contained execution:
   model = nn.Sequential()
end

nout = 10

print "Defining loss function"

if opt.loss == 'nll' then
	-- Log SoftMax loss exp(Wxi + bi) / sum(exp(Wxj + bj)) 
	model:add(nn.LogSoftMax())
	criterion = nn.ClassNLLCriterion()

elseif opt.loss == 'margin' then
	-- SVM loss
	criterion = nn.MultiMarginCriterion()

elseif opt.loss == 'mse' then
	-- MSE loss (y - Wx)^2
	-- for MSE, we add a tanh, to restrict the model's output
    model:add(nn.Tanh())
    criterion = nn.MSECriterion()
    criterion.sizeAverage = false

    if trainData then
      -- convert training labels:
      local trsize = (#trainData.labels)[1]
      local trlabels = torch.Tensor( trsize, noutputs )
      trlabels:fill(-1)
      for i = 1,trsize do
         trlabels[{ i,trainData.labels[i] }] = 1
      end
      trainData.labels = trlabels

      -- convert test labels
      local tesize = (#testData.labels)[1]
      local telabels = torch.Tensor( tesize, noutputs )
      telabels:fill(-1)
      for i = 1,tesize do
         telabels[{ i,testData.labels[i] }] = 1
      end
      testData.labels = telabels
   end
end

print "Loss function ---------------------------"
print (criterion)