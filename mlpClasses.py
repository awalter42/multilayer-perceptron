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
		self.layers = layers + [2]
		self.func = func
		self.learning_rate = learning_rate
		self.weights = self.generateWeights([nb_inputs] + layers)
		self.bias = self.generateBias(layers)
		print(self.bias)


	def sigmoid(self, x):
		return 1 / (1 + math.exp(x))


	def hyperbolicTangent(self, x):
		return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


	def softmax(self, out_vect):
		div = sum(math.exp(out_vect[i]) for i in range(len(out_vect)))

		result = []
		for i in range(len(out_vect)):
			result.append(math.exp(out_vect[i])/ div)
		return result


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
