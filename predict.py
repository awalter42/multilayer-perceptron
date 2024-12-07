import numpy as np
from mlpClasses import Model


if __name__ == '__main__':

	model = Model(file="ModelInfos")

	while True:
		vals = input("values: ")
		if vals == 'exit':
			exit()
		values = vals.split(',')
		for i in range(len(values)):
			values[i] = float(values[i])

		np_values = np.array(values)
		prediction = model.predict(np_values)

		if prediction[0] > prediction[1]:
			print(f"the model predicts malignent with a probability of {prediction[0]}")
		else:
			print(f"the model predicts benign with a probability of {prediction[1]}")
