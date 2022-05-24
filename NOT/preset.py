from ../model.py import *

net = NeuralNetwork(1)
net.setweights(1, [-1])
print(net.test(np.array([1]), 0.5))
print(net.test(np.array([0]), 0.5))
