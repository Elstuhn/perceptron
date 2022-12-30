from ../model.py import *

net = NeuralNetwork(2)
net.setweights(-1.5, [1, 1])
print(net.test(np.array([1, 1])))
print(net.test(np.array([1, 0])))
print(net.test(np.array([0, 1])))
print(net.test(np.array([0, 0])))
