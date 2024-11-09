import sys
import argparse
import random
from mlpClasses import Model
import numpy as np
from statistics import mean


def fetchData(file):
	try:
		file = open(file, "r")
	except:
		sys.exit()

	data = []
	for line in file:
		line = line.replace('M', '1')
		line = line.replace('B', '0')
		splitted = line.split(',')[1:]
		splitted = [eval(v) for v in splitted]
		data.append(np.array(splitted))
	file.close()
	return data


norm_vals = []
for i in range(30):	
	norm_vals.append([0,0])


def standardize(data):
	global norm_vals

	newData = []
	for i in range(len(data)):
		newData.append([data[i][0]])

	for i in range(len(norm_vals)):
		vals = [r[i + 1] for r in data]
		if norm_vals[i] == [0,0]:
			norm_vals[i] = [mean(vals), np.std(vals)]
		for v in range(len(newData)):
			newData[v].append((vals[v] - norm_vals[i][0]) / norm_vals[i][1])

	return newData


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--layer', nargs='*', type=int, default=[10, 10], required=False)
	parser.add_argument('-e', '--epochs', type=int, default=10, required=False)
	parser.add_argument('-L', '--loss', type=str, default='binaryCrossentropy', required=False)
	parser.add_argument('-f', '--func', type=str, default='sigmoid', required=False)
	parser.add_argument('-r', '--learning_rate', type=float, default=1, required=False)
	parser.add_argument('-b', '--batch', type=int, default=10, required=False)

	parser.add_argument('-s', '--seed', type=int, default=None)

	args = parser.parse_args()

	random.seed(args.seed)

	rawTrainingData = fetchData('trainingData.csv')
	rawValidData = fetchData('validationData.csv')

	cleanTrainingData = standardize(rawTrainingData)
	cleanValidData = standardize(rawValidData)
	nb_inputs = len(cleanTrainingData[0]) - 1

	# model = Model(nb_inputs=nb_inputs, layers=args.layer, learning_rate=args.learning_rate)
	# model.fit(cleanTrainingData, cleanValidData, args.func, args.loss, args.batch, args.epochs)

	model = Model(file="ModelInfos")
	test = np.array([0.8351055797139435,0.21873927342986912,0.7444215192881284,0.7224855898592972,-0.6140936307244736,-0.5720718141077551,-0.22520185190357506,0.15080869146071885,0.7970984045493,-1.5225664903295593,0.6498352215251083,-0.6560444714160424,0.5940380969513607,0.6051181471301044,-0.6681653036922511,-0.5727590662765413,-0.4437413067575694,-0.07819782894497812,0.27211745100639445,-0.4746234162212362,0.75679218111701,-0.11174930764398233,0.6978751419370464,0.607074863558349,-0.917077087713495,-0.6509918790143895,-0.3612910157644187,-0.0544925972713536,0.6053335109032636,-1.0588924418664907])
	model.predict(test)

