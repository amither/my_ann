import numpy as np
import random

class Network(object):
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [ np.random.randn(y,x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biaes = [np.random.randn(y,1) for y in sizes[1:] ]

    def SGD(self, training_data, epochs, eta, mini_batch_size):
        n = len(training_data)
        for i in xrange(epochs):
            random.shuffle(training_data)
            mini_batchs = [training_data[j:j+mini_batch_size] for j in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batchs:
                self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biaes]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_b, delta_w = self.backprop( x, y)
            nabla_b = [ b + nb for b, nb in zip(self.biaes, delta_b)]
            nabla_w = [ w + nw for w, nw in zip(self.weights, delta_w)]

        self.biaes = [ b - (eta/len(mini_batch))*nb for b,nb in zip(self.biaes, nabla_b)]
        self.weights = [ w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self,x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biaes]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        zs = []
        a = x
        activations = [x]
        for  b, w in zip(self.biaes, self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmod(z)
            activations.append(a)

        delta = self.cost_derivation(activations[-1], y) * sigmod_prime(zs[-1])
        nabla_b[-1]  = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmod_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    def cost_derivation(a, y):
        return (a-y)

def sigmod(z):
    return 1.0/(1.0+np.exp(-z))

def sigmod_prime(z):
    return sigmod(z) * (1-sigmod(z))




