from ../model.py import *

net = NeuralNetwork(1)
net.setweights(.5, [-1])
print(net.test(np.array([1])))
print(net.test(np.array([0])))
