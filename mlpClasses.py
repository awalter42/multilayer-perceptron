import sys
import random
import math



class Layer:

	def __init__(self, previous=None, output=False):
		self.isOutput = output
		self.values = []
		self.previous = previous
		self.next = None
		self.weight = None
		self.bias = None


	def printLayers(self):
		for w in self.weight:
			print(w)
		print()
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




class Model:

	def __init__(self, nb_inputs, layers, func, learning_rate):
		self.layers = [nb_inputs] + layers + [2]
		self.func = func
		self.learning_rate = learning_rate
		self.weights = self.generateWeights(self.layers)
		self.bias = self.generateBias(self.layers)
		self.expectedOutputs = []
		self.predictedOutputs = []
		self.setupLayers()
		self.inputLayer.printLayers()


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
		for i in range(len(layers)):
			tmp = []
			for j in range(layers[i]):
				tmp.append(round(random.uniform(-1.0, 1.0), 3))
			bias.append(tmp)
		return bias


	def sigmoid(self, x):
		return 1 / (1 + math.exp(-x))


	def derivSigmoid(self, x):
		sigm = sigmoid(x)
		return	sigm * (1 - sigm)


	def hyperbolicTangent(self, x):
		return (2 / (1 + math.exp(-2 * x)) ) - 1


	def derivHyperbolic(self, x):
		return 1 - hyperbolicTangent(x)**2


	def softmax(self, out_vect):
		div = sum(math.exp(out_vect[i]) for i in range(len(out_vect)))

		result = []
		for i in range(len(out_vect)):
			result.append(math.exp(out_vect[i])/ div)
		return result


	def binaryCrossEntropy(self, expectedOutputs, predictedOutputs):
		l = []
		for i in range(len(expectedOutputs)):
			calc = expectedOutputs[i] * math.log(predictedOutputs[i])
			calc += (1 - expectedOutputs[i]) * math.log(1 - predictedOutputs[i])
			l.append(calc)

		return mean(l)


	def derivBCE(self, expected, predicted):
		return -1 * ((expected / predicted) - ((1-expected) / (1-predicted)))


	# def feedForward(self, data, train=False):



	def fit(self, dataTrain, dataValid, loss, batch, epoch):
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
					self.updateWeightBias()
				firstIter = False

				expected = dataTrain[j][0]
				prediction = self.feedForward(dataTrain[j][1:], train=True)
				trainPredictions.append(prediction)
				trainExpected.append(expected)

			self.updateWeightBias()
			validationPrediction, validationExpected = self.validate(dataValid)

			trainLossHistory = self.binaryCrossEntropy(trainPredictions, trainExpected)
			validationLossHistory = self.binaryCrossEntropy(validationPrediction, validationExpected)

			trainAccuracyHistory = self.getAccuracy(trainPredictions, trainExpected)
			validationAccuracyHistory = self.getAccuracy(validationPredictions, validationExpected)



# TODO

# feedForward -> predi int
# calculateGradients :(
# validate -> predi list, expected list
# updateWeightBias -> None
# getAccuracy -> accuracy int


