import argparse
import random


def fetchData(file):
	try:
		file = open(file, "r")
		# file.readline()
	except:
		sys.exit()

	data = []
	for line in file:
		line = line.replace('M', '1')
		line = line.replace('B', '0')
		splitted = line.split(',')[1:]
		splitted = [eval(v) for v in splitted]
		data.append(splitted)
	file.close()
	return data


norm_vals = []
for i in range(30):	
	norm_vals.append([0,0])


def normalize(data):
	global norm_vals

	newData = []
	for i in range(len(data)):
		newData.append([])

	for i in range(len(norm_vals)):
		vals = [r[i + 1] for r in data]
		if norm_vals[i] == [0,0]:
			norm_vals[i] = [min(vals), max(vals)]
		for v in range(len(newData)):
			newData[v].append((vals[v] - norm_vals[i][0]) / (norm_vals[i][1] - norm_vals[i][0]))

	return newData


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--layer', nargs='+', type=int, default=[10, 10, 10], required=False)
	parser.add_argument('-e', '--epochs', type=int, default=50, required=False)
	parser.add_argument('-L', '--loss', type=str, default='binaryCrossentropy', required=False)
	parser.add_argument('-f', '--func', type=str, default='sigmoid', required=False)
	parser.add_argument('-r', '--learning_rate', type=float, default=0.01, required=False)
	parser.add_argument('-b', '--batch', type=int, default=10, required=False)

	parser.add_argument('-s', '--seed', type=int, default=None)

	args = parser.parse_args()

	random.seed(args.seed)

	rawTrainingData = fetchData('trainingData.csv')
	rawValidData = fetchData('validationData.csv')

	cleanTrainingData = normalize(rawTrainingData)
	cleanValidData = normalize(rawValidData)

