import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.distributions as dist

class NeuralNetwork:
    def __init__(self, sizes, activation=torch.sigmoid, alpha=0.5):
        self.num_layers = len(sizes) - 1
        self.theta = []
        self.a = [None] * len(sizes)
        self.a_hat = [None] * len(sizes)
        self.z = [None] * len(sizes)
        self.activation = activation
        self.alpha = 0.03
        self.delta = [None] * len(sizes)
        self.dE_dTheta = [None] * (len(sizes) - 1)
        
        # randomly initalize weights and biases
        stack = []
        for i in range(1, len(sizes)):
            size = sizes[i]
            last_size = sizes[i - 1]

            # mean = 0
            # stddev = 1. / math.sqrt(size)
            # normal_dist = dist.Normal(mean, stddev)

            # stack.append(normal_dist.sample((last_size, size)))
            # stack[-1] = torch.cat((normal_dist.sample((1, 1)).expand(1, size), stack[-1]))
            # stack[-1] = torch.cat((torch.zeros(1, size), stack[-1]))

            # xavier normal gave best percentage of successful convergence for XOR at 84%
            stack.append(init.xavier_normal_(torch.empty(last_size, size)))
            bias = init.xavier_normal_(torch.empty(1, 1)).expand(1, size)
            stack[-1] = torch.cat((bias, stack[-1]))

            # stack.append(init.kaiming_uniform_(torch.empty(last_size, size), a=math.sqrt(5)))
            # fan_in, _ = init._calculate_fan_in_and_fan_out(stack[-1])
            # bound = 1. / math.sqrt(fan_in)
            # bias = init.uniform_(torch.empty(1, size), -bound, bound)
            # stack[-1] = torch.cat((bias, stack[-1]))


        self.theta = stack
    
    def getLayer(self, index):
        return self.theta[index]

    def setLayer(self, index, weights):
        self.theta[index] = weights

    def forward(self, x):
        self.a[0] = x
        for i, t in enumerate(self.theta):
            # prepend bias
            self.a_hat[i] = torch.cat([torch.ones(x.shape[0], 1), self.a[i]], dim=1)
            self.z[i + 1] = torch.mm(self.a_hat[i], t)
            self.a[i + 1] = self.activation(self.z[i + 1])

        return self.a[-1]
    
    # where a is the sigmoid function
    @staticmethod
    def dSigmoid(a):
        return a * (1 - a)

    def backward(self, y):
        # N = number of samples
        N = self.a[0].shape[0]

        # shouldn't use dSigmoid if self.activation isn't sigmoid...
        dA_dZ = self.dSigmoid(self.a[-1])
        dE_dA = -1 * (y - self.a[-1])
        dE_dZ = dA_dZ * dE_dA
        self.delta[-1] = dE_dZ

        # minus 2 because the last layer's calculations have been done above
        for i in range(len(self.delta) - 2, -1, -1):
            # have to remove bias from theta
            dE_dA = torch.mm(self.delta[i + 1], self.theta[i][1:, :].t())
            dA_dZ = self.dSigmoid(self.a[i])

            # is basically dA_dZ * dZ_dZ = dA_dZ -> dA_dZ * dE_dA = dE_dZ
            # should be N * s_(l) dimensions
            # dE_dZ
            self.delta[i] = dE_dA * dA_dZ

            # dE_dZ * dZ_dTheta (because a_hat is output of previous layer)
            # we use a_hat because we want the 1 row for the bias
            # dims should be (s_(l) + 1) * s_(l + 1)
            # because there are N samples, we need to divide the error by N
            self.dE_dTheta[i] = torch.mm(self.a_hat[i].t(), self.delta[i + 1]) / N
        
    # update weights based on your learning rate alpha
    def updateParams(self, alpha=None):
        if alpha is None:
            alpha = self.alpha

        for i, e in enumerate(self.dE_dTheta):
            self.theta[i] = self.theta[i] - (alpha * e)
    
    # calculate MSE loss
    def mse(self, X, Y):
        # our network's solution
        Y_hat = self.forward(X)

        # first subtracts element wise from labels
        # then squares element wise
        # then reduces over columnns so that the dims become N * 1
        se = torch.sum((Y - Y_hat) ** 2, dim=1, keepdim=True)

        # then we sum rows and divide by number of rows, N
        mse = (1. / Y_hat.shape[0]) * torch.sum(se)

        return mse
