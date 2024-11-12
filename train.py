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
	# model.fit(cleanTrainingData, cleanValidData, args.func, args.loss, args.batch, args.epochs, stan_vals)

	model = Model(file="ModelInfos")
	test = np.array([19.19,15.94,126.3,1157,0.08694,0.1185,0.1193,0.09667,0.1741,0.05176,1,0.6336,6.971,119.3,0.009406,0.03055,0.04344,0.02794,0.03156,0.003362,22.03,17.81,146.6,1495,0.1124,0.2016,0.2264,0.1777,0.2443,0.06251])
	model.predict(test)

