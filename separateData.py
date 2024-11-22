import random


def fetchData():
	try:
		file = open('data.csv', "r")
	except:
		print(f'There has been a problem when fetching the file {file}')
	tab = []
	for line in file:
		tab.append(line)
	file.close()
	return tab


def splitData(data):
	length = len(data)
	trainSize = int(length*0.85)
	training = data[:trainSize]
	validation = data[trainSize:]

	return training, validation


def saveDatas(training, validation):
	try:
		f = open('trainingData.csv', 'w')
		for line in training:
			f.write(line)
		f.close()

		f = open('validationData.csv', 'w')
		for line in validation:
			f.write(line)
		f.close()
	except:
		print('there has been a problem saving the data')


if __name__ == '__main__':
	data = fetchData()
	random.shuffle(data)
	training, validation= splitData(data)
	saveDatas(training, validation)

