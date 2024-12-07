Multilayer Perceptron
==========================
A binary classifier of breast cancer cells

This is a python implementation of a Neural Network using numpy

Usage
===========================

**Separate the data:**
```bash
python3 separateData.py
```

**train the model:**
```bash
python3 train.py [-l] {layer sizes (i.e 5 5)} [-e] {number of epochs} [-f] {activation function (sigmoid or tanh)} [-r] {learning rate (float usually beetween 0 and 1)} [-b] {batch size} [-s] {random seed}
```
All options have default values so `python3 train.py` would work

two graphs representing the Loss and Accuracy training on both training set and validation set will be shown

You will then have the option to save the models informations to be used on a prediction program

**predict from the saved model:**
```bash
python3 predict.py
```
You will be given a prompt in an infinite loop. Each iterations you will be able to give cells values, and the predicted state will be given in the terminal

