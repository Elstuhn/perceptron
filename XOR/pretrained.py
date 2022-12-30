import random 
import numpy as np
import typing as t
from tqdm import tqdm

tensor = t.Union[list, np.ndarray]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def unitstep(x):
    if 0 > x:
        return 0
    else:
        return 1
    
def test_sample(net, n_features, num_sample = 10, operation : str = "or"):
    """
    operation denotes the operation that is being tested e.g AND OR XOR NOR
    """
    for i in range(num_sample):
        nums = [random.randint(0, 1) for i in range(n_features)]
        prediction = net.test(np.array(nums))
        print(f"{(' '+operation+' ').join([str(num) for num in nums])}: {prediction}")
    
def gen_sample(net, n_features, num_sample = 10):
    for i in range(num_sample):
        nums = [random.randint(0, 1) for i in range(n_features)]
        prediction = net.test(np.array(nums))
        print(f"{' or '.join([str(num) for num in nums])}: {prediction}")
        yield nums + [prediction]
        
def xor(*numbers):
    return False if (not sum(numbers[0])) else True

def compare(num1, num2):
    return True if not (num1-num2) else False

def checkaccuracy(net, n_features, test_samples : int):
        samples = gen_sample(net, n_features, test_samples)
        correct = total = 0
        for numbers in samples:
            correct += compare(xor(numbers[:-1]), (lambda x : 0 if x < 0.5 else 1)(numbers[-1]))
            total += 1
        return correct/total

class Node:
    def __init__(self, value : float = None, weight = "random"):
        if weight == "random" : self.weight = 2 * random.random() - 1
        else : self.weight = weight
        self.value = value
        
    def multiplied(self):
        return self.value * self.weight
        
class NeuralNetwork:
    """
    X will be (batch num, n_features)
    """
    def __init__(self, numNodes : int, weights : tensor = "random", bias : float = "random"):
        if weights == "random":
            self.inputlayer = np.array([Node() for i in range(numNodes)])
        else:
            self.inputlayer = np.array([Node(weight = weights[i]) for i in range(numNodes)])
        self.bias = random.random() if bias == "random" else bias
        self.activation = unitstep
        self.numNodes = numNodes
        
    def output(self):
        return self.activation(sum([i.multiplied() for i in self.inputlayer]) + self.bias)
    
    def backprop(self, yhat, x, y, lr):
        for i in range(len(self.inputlayer)):
            self.inputlayer[i].weight += lr*((y-yhat)*x[i])
            self.bias += lr*(y-yhat)
        #print(f"W1: {self.inputlayer[0].weight} W2: {self.inputlayer[1].weight} W3: {self.inputlayer[2].weight} bias: {self.weightbias}")
    
    def train(self, X : tensor, y : tensor, epochs : int, lr = 0.005, returnweights = False, autostop = 0):
        lastAcc = curAcc = 0
        count = 0
        self.changeNeurons(X)
        for epoch in tqdm(range(epochs)):
            #print(f"Epoch {epoch+1}")
            for ind in range(len(X)):
                self.changeNeurons(X[ind])
                x_i = np.insert(X[ind], self.numNodes, self.bias).reshape(-1,1)
                y_hat = self.output()
                self.backprop(y_hat, x_i, y[ind], lr)
                
            if autostop:
                if not (count-autostop):
                    count = 0
                    lastAcc = curAcc
                    curAcc = checkaccuracy(self, self.numNodes, 100)
                    print(f"Last accuracy: {lastAcc}\nNew Accuracy: {curAcc}")
                    if (curAcc <= lastAcc):
                        break
            count += 1
                
        if returnweights: return [self.bias] + [node.weight for node in self.inputlayer]
    
    
    def test(self, X : tensor):
        self.changeNeurons(X)
        try:
            return round(self.output()[0], 2)
        except:
            return round(self.output(), 2)
        
          
    def changeNeurons(self, X):
        for i in range(len(self.inputlayer)):
            self.inputlayer[i].value = X[i]
            
    def setweights(self, bias : float, *weights : tensor):
        self.bias = bias 
        for i in range(len(*weights)):
            self.inputlayer[i].weight = weights[0][i]

def XOR(x : tensor):
    NOT = NeuralNetwork(1, [-1], 0.5)
    AND = NeuralNetwork(2, [1, 1], -1.5)
    OR = NeuralNetwork(2, [1, 1], -.5)
    inputs = np.array(x)
    op1 = AND.test(inputs)
    op2 = NOT.test(np.array([op1]))
    op3 = OR.test(inputs)
    finalop = AND.test(np.array([op2, op3]))
    return finalop


print(XOR([1, 1]))
print(XOR([1, 0]))
print(XOR([0, 0]))
print(XOR([0, 1]))
