require 'torch'
require 'csvigo'

local separator = ","

if not testData then
	test_raw = torch.load('data/test_32x32.t7', 'ascii')
	testData = {
		data = test_raw.X:transpose(3, 4),
		y = test_raw.y[1],
		size = function() return (#testData.data)[1] end
	}	
end

if model then
	testData.y = model:forward(testData.data)
end

local csvfile = csvigo.File("output/submission.csv", "w", separator)

-- Write header
csvfile:write({"Id", "Prediction"})

for i = 1, testData:size() do
	csvfile:write({i, testData.y[i]})
end

csvfile:close()