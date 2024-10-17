import sys
import random
import math


def sigmoid(x):
	return 1 / (1 + math.exp(-x))


def derivSigmoid(x):
	sigm = sigmoid(x)
	return	sigm * (1 - sigm)


def hyperbolicTangent(x):
	return (2 / (1 + math.exp(-2 * x)) ) - 1


def derivHyperbolic(x):
	return 1 - hyperbolicTangent(x)**2


def softmax(out_vect):
	div = sum(math.exp(out_vect[i]) for i in range(len(out_vect)))

	result = []
	for i in range(len(out_vect)):
		result.append(math.exp(out_vect[i])/ div)
	return result


def binaryCrossEntropy(expectedOutputs, predictedOutputs):
	l = []
	for i in range(len(expectedOutputs)):
		calc = expectedOutputs[i] * math.log(predictedOutputs[i])
		calc += (1 - expectedOutputs[i]) * math.log(1 - predictedOutputs[i])
		l.append(calc)

	return mean(l)


def derivBCE(expected, predicted):
	return -1 * ((expected / predicted) - ((1-expected) / (1-predicted)))


class Layer:

	def __init__(self, previous=None, output=False):
		self.isOutput = output
		self.values = []
		self.previous = previous
		self.next = None
		self.weight = None
		self.bias = None


	def printLayers(self):
		print("weights:")
		for w in self.weight:
			print(w)
		print("\nbias:")
		print(self.bias, '\n', '\n')
		if not self.next.isOutput:
			self.next.printLayers()


	def addValue(self, v):
		if self.values == []:
			self.size = len(v)
		self.values.append(v)


	def setNext(self, next):
		self.next = next


	def setWeight(self, weight):
		self.weight = weight


	def setBias(self, bias):
		self.bias = bias


	def activate(self, output, func):
		f = func.lower()
		if f == 'sigmoid':
			for i in range(len(output)):
				output[i] = sigmoid(output[i])
		elif f == 'hyperbolicTangent' or f == 'tanh':
			for i in range(len(output)):
				output[i] = hyperbolicTangent(output[i])
		return output



	def feedForward(self, input_list, func, loss, expect, train=False):
		if self.isOutput == True:
			output = softmax(input_list)
			return output
		else:
			output = []
			for _ in range(len(self.weight[0])):
				output.append(0)

			for i in range(len(self.weight)):
				for j in range(len(self.weight[i])):
					output[j] += input_list[i] * self.weight[i][j]

			for i in range(len(self.bias)):
				output[i] += self.bias[i]

			output = self.activate(output, func)
			out = self.next.feedForward(output, func, loss, expect, train=train)

		return out



class Model:

	def __init__(self, nb_inputs, layers, learning_rate):
		self.layers = [nb_inputs] + layers + [2]
		self.learning_rate = learning_rate
		self.weights = self.generateWeights(self.layers)
		self.bias = self.generateBias(self.layers)
		self.expectedOutputs = []
		self.predictedOutputs = []
		self.setupLayers()
		# self.inputLayer.printLayers()


	def setupLayers(self):
		self.inputLayer = Layer()
		self.inputLayer.setWeight(self.weights[0])
		self.inputLayer.setBias(self.bias[0])
		prev = self.inputLayer
		current = None
		for i in range(1, len(self.layers) - 1):
			current = Layer(previous=prev)
			current.setWeight(self.weights[i])
			current.setBias(self.bias[i])
			prev.setNext(current)
			prev = current
		self.outputLayer = Layer(previous=prev, output=True)
		prev.setNext(self.outputLayer)



	def generateWeights(self, layers):
		weights = []
		for i in range(len(layers) - 1):
			weight_li = []
			for l in range(layers[i]):
				weight_li.append([])
				for j in range(layers[i + 1]):
					weight_li[l].append(round(random.uniform(-10.0, 10.0), 3))
			weights.append(weight_li)
		return weights


	def generateBias(self, layers):
		bias = []
		for i in range(1, len(layers)):
			tmp = []
			for j in range(layers[i]):
				tmp.append(round(random.uniform(-1.0, 1.0), 3))
			bias.append(tmp)
		return bias


	def fit(self, dataTrain, dataValid, func, loss, batch, epoch):
		trainLossHistory = []
		trainAccuracyHistory = []
		validationLossHistory = []
		validationAccuracyHistory = []
		for _ in range(epoch):
			trainPredictions = []
			trainExpected = []
			firstIter = True
			for j in range(len(dataTrain)):

				if j % batch == 0 and not firstIter:
					sys.exit()
					self.updateWeightBias()
				firstIter = False

				expected = dataTrain[j][0]

				expect = [1, 1]
				expect[expected] -= 1
				prediction = self.inputLayer.feedForward(dataTrain[j][1:], func, loss, expect, train=True)
				# print(prediction, expect)
				trainPredictions.append(prediction)
				trainExpected.append(expected)
			sys.exit()


			self.updateWeightBias()
			validationPrediction, validationExpected = self.validate(dataValid)

			trainLossHistory = self.binaryCrossEntropy(trainPredictions, trainExpected)
			validationLossHistory = self.binaryCrossEntropy(validationPrediction, validationExpected)

			trainAccuracyHistory = self.getAccuracy(trainPredictions, trainExpected)
			validationAccuracyHistory = self.getAccuracy(validationPredictions, validationExpected)



# TODO

# calculateGradients :(
# validate -> predi list, expected list
# updateWeightBias -> None
# getAccuracy -> accuracy int


