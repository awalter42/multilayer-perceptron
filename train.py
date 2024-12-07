import sys
import argparse
import random
from mlpClasses import Model
import numpy as np
from statistics import mean


def fetchData(file_name):
	try:
		file = open(file_name, "r")
	except:
		print(f"problem fetching data from {file_name}")
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
	parser.add_argument('-l', '--layer', nargs='+', type=int, default=[5, 5], required=False)
	parser.add_argument('-e', '--epochs', type=int, default=50, required=False)
	parser.add_argument('-f', '--func', type=str, default='sigmoid', required=False)
	parser.add_argument('-r', '--learning_rate', type=float, default=0.2, required=False)
	parser.add_argument('-b', '--batch', type=int, default=20, required=False)

	parser.add_argument('-s', '--seed', type=int, default=None)

	args = parser.parse_args()

	if args.epochs <= 0:
		print('epochs must be at least 1')
		exit()
	args.func = args.func.lower()
	if args.func not in ['sigmoid', 'tanh', 'hyperbolictangent']:
		print('this activation function is not supported')
		exit()
	if args.learning_rate <= 0:
		print('learning rate must be above 0')
		exit()
	if args.batch <= 0:
		print('batch size must be at least 1')
		exit()


	random.seed(args.seed)

	rawTrainingData = fetchData('trainingData.csv')
	rawValidData = fetchData('validationData.csv')

	cleanTrainingData = standardize(rawTrainingData)
	cleanValidData = standardize(rawValidData)
	nb_inputs = len(cleanTrainingData[0]) - 1

	model = Model(nb_inputs=nb_inputs, layers=args.layer, learning_rate=args.learning_rate)
	model.fit(cleanTrainingData, cleanValidData, args.func, 'binaryCrossentropy', args.batch, args.epochs, stan_vals)
