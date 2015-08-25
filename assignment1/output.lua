require 'torch'
require 'csvigo'
require 'nn'

if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Saving to Output')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-type', 'double', 'type: double | float | cuda')
   cmd:text()
   opt = cmd:parse(arg or {})
end

local separator = ","

if not testData then
	test_raw = torch.load('data/test_32x32.t7', 'ascii')
	testData = {
		data = test_raw.X:transpose(3, 4),
		y = test_raw.y[1],
		size = function() return (#testData.data)[1] end
	}	
end

local model_path = paths	.concat(opt.save, "model.net")
model = torch.load(model_path)

local csvfile = csvigo.File("output/submission.csv", "w", separator)

-- Write header
csvfile:write({"Id", "Prediction"})

-- Predict and write data
for i = 1, testData:size() do
	if model then
		local input = testData.data[i]
		if opt.type == 'double' then input = input:double()
		elseif opt.type == 'cuda' then input = input:cuda()
		end
		testData.y[i] = model:forward(input)
	end
	csvfile:write({i, testData.y[i]})
end

csvfile:close()