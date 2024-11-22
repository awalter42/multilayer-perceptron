Multilayer Perceptron
==========================
A binary classifier of breast cancer cells

This is a python implementation of a Neural Network using numpy

Usage
===========================

Separate the data:
```bash
python3 separateData.py
```

train the model:
```bash
python3 train.py [-l] {layer sizes (i.e 5 5)} [-e] {number of epochs} [-f] {activation function (sigmoid or tanh)} [-r] {learning rate (float usually beetween 0 and 1)} [-b] {batch size} [-s] {random seed}
```
All options have default values so `python3 train.py` would work

You will then have the option to save the models informations to be used on a prediction program

**Coming Soon**
- predictions
- plots of the model's training
