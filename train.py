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


stan_vals = []
for i in range(30):	
	stan_vals.append([0,0])


def standardize(data):
	global stan_vals

	newData = []
	for i in range(len(data)):
		newData.append([data[i][0]])

	for i in range(len(stan_vals)):
		vals = [r[i + 1] for r in data]
		if stan_vals[i] == [0,0]:
			stan_vals[i] = [mean(vals), np.std(vals)]
		for v in range(len(newData)):
			newData[v].append((vals[v] - stan_vals[i][0]) / stan_vals[i][1])

	return newData


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--layer', nargs='*', type=int, default=[5, 5], required=False)
	parser.add_argument('-e', '--epochs', type=int, default=50, required=False)
	# parser.add_argument('-L', '--loss', type=str, default='binaryCrossentropy', required=False)
	parser.add_argument('-f', '--func', type=str, default='sigmoid', required=False)
	parser.add_argument('-r', '--learning_rate', type=float, default=0.2, required=False)
	parser.add_argument('-b', '--batch', type=int, default=20, required=False)

	parser.add_argument('-s', '--seed', type=int, default=None)

	args = parser.parse_args()

	random.seed(args.seed)

	rawTrainingData = fetchData('trainingData.csv')
	rawValidData = fetchData('validationData.csv')

	cleanTrainingData = standardize(rawTrainingData)
	cleanValidData = standardize(rawValidData)
	nb_inputs = len(cleanTrainingData[0]) - 1

	model = Model(nb_inputs=nb_inputs, layers=args.layer, learning_rate=args.learning_rate)
	model.fit(cleanTrainingData, cleanValidData, args.func, 'binaryCrossentropy', args.batch, args.epochs, stan_vals)

	# model = Model(file="ModelInfos")
	# test = np.array([14.99,25.2,95.54,698.8,0.09387,0.05131,0.02398,0.02899,0.1565,0.05504,1.214,2.188,8.077,106,0.006883,0.01094,0.01818,0.01917,0.007882,0.001754,14.99,25.2,95.54,698.8,0.09387,0.05131,0.02398,0.02899,0.1565,0.05504])
	# model.predict(test)
