---
-- Initial require section. 
---
require 'torch'
require 'image'
require 'nn'

---
-- Read t7 file. Don't forget 'asci'' argument to load method
---
print "Reading train dataset: "
train_raw = torch.load('data/train_32x32.t7', 'ascii')
print (train_raw)

--- 
-- We need to rearrange the data in format recognized by torch
-- X is a 4D-tensor. 73257x3x32x32. First dimension is the number of samples,
-- next is the number of channels and finally we have 32x32 images.We first take its transpose.
---
trainData = {
	data = train_raw.X:transpose(3, 4),
	labels = train_raw.y[1],
	size = function() return (#trainData.data)[1] end
}
print ("Training Data = ", trainData)

---
-- Need to perform similar transformation of test data
---
print "Reading test dataset: "
test_raw = torch.load('data/test_32x32.t7', 'ascii')
print (test_raw)

---
-- Perform similar transformation on test data
---
testData = {
	data = test_raw.X:transpose(3, 4),
	labels = test_raw.y[1],
	size = function() return (#testData.data)[1] end
}
print ("Test Data = ", testData)

-- Define additional parameters used in later files
trsize = trainData:size()
print ("trsize = ", trsize)

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes). Types can be easily converted in Torch, 
-- in general by doing: dst = src:type('torch.TypeTensor'), 
-- where Type=='Float','Double','Byte','Int',... Shortcuts are provided
-- for simplicity (float(),double(),cuda(),...):
print "Start Data Transformation -----------------------------------"
print "Converting data from bytes => float"
trainData.data = trainData.data:float()
testData.data = testData.data:float()

---
-- The original data is in RGB space, we convert it yo YUV using rgb2yuv
---
print "Converting Train Data from RGB -> YUV"
for i = 1,trainData:size() do
	trainData.data[i] = image.rgb2yuv(trainData.data[i])
end
print "Converting Test Data from RGB -> YUV"
for i = 1,testData:size() do
	testData.data[i] = image.rgb2yuv(testData.data[i])
end

-- Name channels for convenience
channels = {'y','u','v'}

---
-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
---
print "Mean Normalizing Data: subtract mean and divide by std (Global Normalization)"
mean = {}
std = {}
for i,channel in ipairs(channels) do
	mean[i] = trainData.data[{ {}, i, {}, {} }]:mean()
	std[i] = trainData.data[{ {}, i, {}, {} }]:std()
	-- First normalize train data
	trainData.data[{ {}, i, {}, {} }]:add(-mean[i])
	trainData.data[{ {}, i, {}, {} }]:div(std[i])
	-- Now normalize test datas
	testData.data[{ {}, i, {}, {} }]:add(-mean[i])
	testData.data[{ {}, i, {}, {} }]:div(std[i])
end

---
-- For each 'y', 'u', 'v' channel, find 1D gaussian gradient around it
-- and use spatial contrastive normalization to locally normalize it
---
print "Using gaussian1D and SpatialContrastiveNormalization to locally normalize data"
neighborhood = image.gaussian1D(13)
normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()
for c in ipairs(channels) do
	-- Training data
	for i = 1,trainData:size() do
		trainData.data[{ i, {c}, {}, {} }] = normalization:forward(trainData.data[{ i, {c}, {}, {} }])
	end
	-- Test data
	for i = 1,testData:size() do
		testData.data[{ i, {c}, {}, {} }] = normalization:forward(testData.data[{ i, {c}, {}, {} }])
	end
end

print "End Data Transformation -----------------------------------"

print "Verifying statistics after Transformation --------------------"
for i,c in ipairs(channels) do
	trainMean = trainData.data[{ {}, i }]:mean()
	trainStd = trainData.data[{ {}, i }]:std()

	testMean = testData.data[{ {}, i }]:mean()
	testStd = testData.data[{ {}, i }]:std()

	print('training data, '..c..'-channel, mean: ' .. trainMean)
   	print('training data, '..c..'-channel, standard deviation: ' .. trainStd)

   	print('test data, '..c..'-channel, mean: ' .. testMean)
   	print('test data, '..c..'-channel, standard deviation: ' .. testStd)
end
print "End Verifying statistics after Transformation --------------------"

---
-- Finally print images. Use itorch.image method. Only works in itorch notebook
---
if itorch then
	print "Visualization -------------------"
	images_y = trainData.data[{ {1,256}, 1 }]
	images_u = trainData.data[{ {1,256}, 2 }]
	images_v = trainData.data[{ {1,256}, 3 }]

	itorch.image(images_y)
	itorch.image(images_u)
	itorch.image(images_v)
end
