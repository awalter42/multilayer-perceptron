import sys
import random
import numpy as np
from statistics import mean

def sigmoid(x):
	if x > 0:
		z = np.exp(-x)
		return float(1/(1+z))
	else:
		z = np.exp(x)
		return float(z/(1+z))


def derivSigmoid(x):
	sigm = sigmoid(x)
	return	sigm * (1 - sigm)


def hyperbolicTangent(x):
	return float((2 / (1 + np.exp(-2 * x)) ) - 1)


def derivHyperbolic(x):
	return 1 - hyperbolicTangent(x)**2


def softmax(out_vect):
	div = sum(np.exp(out) for out in out_vect)

	result = []
	for out in out_vect:
		result.append(float(np.exp(out)/ div))
	return result


def binaryCrossEntropy(expectedOutputs, predictedOutputs):
	l = []
	for i in range(len(expectedOutputs)):
		calc = expectedOutputs[i] * np.log(predictedOutputs[i])
		calc += (1 - expectedOutputs[i]) * np.log(1 - predictedOutputs[i])
		l.append(calc)

	return -1 * mean(l)


class Layer:

	def __init__(self, previous=None, output=False):
		self.isOutput = output
		self.values = []
		self.previous = previous
		self.next = None
		self.weight = None
		self.bias = None
		self.gradWeight = []
		self.gradBias = []


	def printLayers(self):
		print("weights:")
		for i in range(len(self.weight)):
			print(self.weight[i])
			print(self.gradWeight[i])

		print("\nbias:")
		print(self.bias)
		print(self.gradBias, '\n', '\n')
		if not self.next.isOutput:
			self.next.printLayers()


	def addValue(self, v):
		self.values.append(v)


	def setNext(self, next):
		self.next = next


	def setWeight(self, weight):
		self.weight = np.array(weight)
		for l in self.weight:
			grad = []
			for i in range(len(l)):
				grad.append([])
			self.gradWeight.append(grad)


	def setBias(self, bias):
		self.bias = np.array(bias)
		for i in range(len(bias)):
			self.gradBias.append([])


	def updateWeightBias(self, learning_rate):
		for i in range(len(self.weight)):
			for j in range(len(self.weight[i])):
				self.weight[i][j] -= learning_rate * mean(self.gradWeight[i][j])
				self.gradWeight[i][j] = []

		for i in range(len(self.bias)):
			self.bias[i] -= learning_rate * mean(self.gradBias[i])
			self.gradBias[i] = []

		if not self.next.isOutput:
			self.next.updateWeightBias(learning_rate)


	def activate(self, output, func):
		f = func.lower()
		activ_output = []
		if f == 'sigmoid':
			for i in range(len(output)):
				activ_output.append(sigmoid(output[i]))
		elif f == 'hyperbolicTangent' or f == 'tanh':
			for i in range(len(output)):
				activ_output.append(hyperbolicTangent(output[i]))
		return activ_output


	def derivActivate(self, output, func):
		f = func.lower()
		derivated = []
		if f == 'sigmoid':
			for i in range(len(output)):
				derivated.append(derivSigmoid(output[i]))
		elif f == 'hyperbolicTangent' or f == 'tanh':
			for i in range(len(output)):
				derivated.append(derivHyperbolic(output[i]))
		return derivated


	def feedForward(self, input_list, z_list, func, loss, expect, train=False):

		self.size = len(input_list)
		self.values = []

		###FORWARD PASS###
		if self.isOutput:
			out = softmax(z_list)
			for value in out:
				self.addValue(value)
		else:
			for value in input_list:
				self.addValue(value)
			output = np.add(np.dot(input_list, self.weight), self.bias)
			activ_output = self.activate(output, func)
			out = self.next.feedForward(activ_output, output, func, loss, expect, train=train)

		###BACKWARD PASS###
		if train:
			if self.isOutput:

				for j in range(self.size):
					calc = 1 * (out[j] - expect[j])
					self.previous.gradBias[j].append(float(calc))

				for i in range(self.previous.size):
					for j in range(self.size):
						prev_activated = self.previous.values[i]
						calc = prev_activated * (out[j] - expect[j])
						self.previous.gradWeight[i][j].append(float(calc))

				gradNeuron = []
				for i in range(self.previous.size):
					tab = []
					for j in range(self.size):
						calc = self.previous.weight[i][j] * (out[j] - expect[j])
						tab.append(float(calc))
					gradNeuron.append(sum(tab))
				self.previous.gradNeuron = gradNeuron

			elif self.previous != None:
				derivated = self.derivActivate(z_list, func)
				for j in range(self.size):
					calc = 1 * derivated[j] * self.gradNeuron[j]
					self.previous.gradBias[j].append(float(calc))

				for i in range(self.previous.size):
					for j in range(self.size):
						prev_activated = self.previous.values[i]
						calc = prev_activated * derivated[j] * self.gradNeuron[j]
						self.previous.gradWeight[i][j].append(float(calc))

				gradNeuron = []
				for i in range(self.previous.size):
					tab = []
					for j in range(self.size):
						calc = self.previous.weight[i][j] * derivated[j] * self.gradNeuron[j]
						tab.append(float(calc))
					gradNeuron.append(sum(tab))
				self.previous.gradNeuron = gradNeuron

		return out



class Model:

	def __init__(self, **kwargs):
		if len(kwargs.keys()) == 3:
			self.layers = [kwargs.get('nb_inputs')] + kwargs.get('layers') + [2]
			self.learning_rate = kwargs.get('learning_rate')
			self.weights = self.generateWeights(self.layers)
			self.bias = self.generateBias(self.layers)
		else:
			self.makeModelFromFile(kwargs.get('file'))

		self.setupLayers()


	def makeModelFromFile(self, file):
		try:
			file = open(file, "r")
			file.readline()
		except:
			print(f'There has been a problem when fetching the file {file}')



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
					weight_li[l].append(random.uniform(-1, 1))
			weights.append(weight_li)
		return weights


	def generateBias(self, layers):
		bias = []
		for i in range(1, len(layers)):
			tmp = []
			for j in range(layers[i]):
				tmp.append(random.uniform(-1, 1))
			bias.append(tmp)
		return bias


	def validate(self, dataValid, func, loss):
		validationProbability = []
		validationPrediction = []
		validationExpected = []
		for line in dataValid:
			expected = line[0]

			expect = [1, 1]
			expect[int(expected)] -= 1
			prediction = self.inputLayer.feedForward(line[1:], [], func, loss, expect, train=False)
			validationProbability.append(prediction[0])
			validationPrediction.append(prediction.index(min(prediction)))
			validationExpected.append(expected)

		return validationPrediction, validationExpected, validationProbability


	def getAccuracy(self, expected, prediction):
		result = []
		for i in range(len(expected)):
			if expected[i] == prediction[i]:
				result.append(1.0)
			else:
				result.append(0.0)
		return mean(result)


	def fit(self, dataTrain, dataValid, func, loss, batch, epoch):
		trainLossHistory = []
		trainAccuracyHistory = []
		validationLossHistory = []
		validationAccuracyHistory = []

		for i in range(epoch):
			trainProbability = []
			trainPredictions = []
			trainExpected = []

			firstIter = True
			random.shuffle(dataTrain)

			for j in range(len(dataTrain)):

				if j % batch == 0 and not firstIter:
					self.inputLayer.updateWeightBias(self.learning_rate)
				firstIter = False

				expected = dataTrain[j][0]
				expect = [1, 1]
				expect[int(expected)] -= 1

				prediction = self.inputLayer.feedForward(dataTrain[j][1:], [], func, loss, expect, train=True)

				trainProbability.append(prediction[0])
				trainPredictions.append(prediction.index(min(prediction)))
				trainExpected.append(expected)

			self.inputLayer.updateWeightBias(self.learning_rate)
			validationPrediction, validationExpected, validationProbability = self.validate(dataValid, func, loss)

			trainLossHistory.append(binaryCrossEntropy(trainExpected, trainProbability))
			validationLossHistory.append(binaryCrossEntropy(validationExpected, validationProbability))
			trainAccuracyHistory.append(self.getAccuracy(trainPredictions, trainExpected))
			validationAccuracyHistory.append(self.getAccuracy(validationPrediction, validationExpected))

			if i > 0:
				if trainLossHistory[-1] < trainLossHistory[-2]:
					self.learning_rate *= 0.95
				elif trainLossHistory[-1] > trainLossHistory[-2]:
					self.learning_rate *= 1.2

			print(f'epoch {"".join(["0" for t in range(len(str(epoch)) - len(str(i+1)))])}{i + 1}/{epoch} - train loss: {round(trainLossHistory[-1], 4)} - valid loss: {round(validationLossHistory[-1], 4)}')
		print(f"\nAccuracy on last epoch:\n Training: {trainAccuracyHistory[-1]}\n validation: {validationAccuracyHistory[-1]}")

		if (input('Do you want to save this model? y/n: ') == 'y'):
			self.saveModel(func)
		else:
			print('ok, not saving :(')


	def saveModel(self, func):
		try:
			save_str = str(self.layers)[1:-1] + '\n\n'
			
			l = self.inputLayer
			while not l.isOutput:
				for i in range(l.size):
					line = ''
					for j in range(l.next.size):
						line += str(l.weight[i][j]) + ','
					line = line[:-1]
					save_str += line + '\n'
				save_str += '\n'
				line = ''
				for j in range(l.next.size):
					line += str(l.bias[j]) + ','
				line = line[:-1]
				save_str += line + '\n\n'
				l = l.next

			save_str += func

			f = open('ModelInfos', 'w')
			f.write(save_str)
			f.close()
		except:
			print('there has been a problem saving the model')
