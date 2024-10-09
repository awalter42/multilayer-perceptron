import sys





class Layer:

	def __init__(self, size, type):
		self.values = []
		self.weights = []
		self.size = size
		self.type = type
		self.next = None


class Network:

	def __init__(self, nbLayer, sizeLayers):
		self.nbLayer = nbLayer
		self.sizeLayers = sizeLayers
		
