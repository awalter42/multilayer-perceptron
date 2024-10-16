import sys
import random
import math



# class Layer:

# 	def __init__(self, size, type):
# 		self.values = []
# 		self.weights = []
# 		self.size = size
# 		self.type = type
# 		self.next = None


class Model:

	def __init__(self, nb_inputs, layers, func, learning_rate):
		self.layers = [nb_inputs] + layers + [2]
		self.func = func
		self.learning_rate = learning_rate
		self.weights = self.generateWeights(self.layers)
		self.bias = self.generateBias(layers)
		self.expectedOutputs = []
		self.predictedOutputs = []


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


	def binaryCrossEntropy(self):
		l = []
		for i in range(len(self.expectedOutputs)):
			calc = self.expectedOutputs[i] * math.log(self.predictedOutputs[i])
			calc += (1 - self.expectedOutputs[i]) * math.log(1 - self.predictedOutputs[i])
			l.append(calc)

		return mean(l)


	def derivBCE(self, expected, predicted):
		return -1 * ((expected / predicted) - ((1-expected) / (1-predicted)))


	def generateWeights(self, layers):
		weights = []
		for i in range(len(layers) - 1):
			weight_li = []
			for l in range(layers[i]):
				weight_li.append([])
				for j in range(layers[i + 1]):
					weight_li[l].append(round(random.uniform(-10.0, 10.0), 3))
			weights.append(weight_li)
		print(weights)
		return weights


	def generateBias(self, layers):
		bias = []
		for i in range(len(layers)):
			tmp = []
			for j in range(layers[i]):
				tmp.append(round(random.uniform(-1.0, 1.0), 3))
			bias.append(tmp)
		return bias
