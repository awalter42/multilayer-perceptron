import sys
import random




# class Layer:

# 	def __init__(self, size, type):
# 		self.values = []
# 		self.weights = []
# 		self.size = size
# 		self.type = type
# 		self.next = None


class Model:

	def __init__(self, nb_inputs, layers, func, learning_rate):
		self.layers = layers
		self.func = func
		self.learning_rate = learning_rate
		self.weights = self.generateWeights([nb_inputs] + layers + [1])
		self.bias = self.generateBias(layers)
		print(self.bias)


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
