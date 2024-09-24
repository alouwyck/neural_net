import numpy as np
import random


g = lambda z: 1 / (1 + np.exp(-z))
dgdz = lambda z: g(z) * (1 - g(z))


class NeuralNetwork:

    def __init__(self, num_of_nodes, alpha, num_of_epochs, batch_size=1, activation=(g, dgdz)):
        self.num_of_nodes = num_of_nodes
        self.nlayers = len(num_of_nodes) + 2
        self.g, self.dgdz = activation[0], activation[1]
        self.alpha = alpha
        self.num_of_epochs = num_of_epochs
        self.batch_size = batch_size
        self.W = None
        self.losses = None

    def __initialize_weights(self, X, Y):
        self.W = []
        n1 = X.shape[1] + 1
        for k in self.num_of_nodes:
            n2 = k
            self.W.append(np.random.randn(n2, n1))
            n1 = k + 1
        self.W.append(np.random.randn(Y.shape[1], n1))

    @staticmethod
    def __add_ones(M):
        return np.concatenate((np.ones((1, M.shape[1])), M), axis=0)

    def __update(self, X, Y):
        Z = []
        A = [self.__add_ones(X.T)]
        for k in range(self.nlayers - 1):
            Z.append(np.dot(self.W[k], A[k]))
            A.append(self.__add_ones(self.g(Z[k])))
        E = A[-1][1:, :] - Y.T
        D = E * self.dgdz(Z[-1])
        for k in range(self.nlayers - 2, 0, -1):
            self.W[k] -= self.alpha * np.dot(D, A[k].T)
            D = np.dot(self.W[k][:, 1:].T, D) * self.dgdz(Z[k-1])
        self.W[0] -= self.alpha * np.dot(D, A[0].T)

    def fit(self, X, Y):
        nsamples = X.shape[0]
        self.__initialize_weights(X, Y)
        self.losses = []
        idx = np.array(range(nsamples))
        for _ in range(self.num_of_epochs):
            random.shuffle(idx)
            b = np.arange(self.batch_size)
            for i in range(nsamples // self.batch_size):
                self.__update(X[idx[b], :], Y[idx[b], :])
                b += self.batch_size
            self.losses.append(self.error(X, Y))
        self.losses = np.asarray(self.losses)

    def predict(self, X):
        A = self.__add_ones(X.T)
        for k in range(self.nlayers - 1):
            Z = np.dot(self.W[k], A)
            A = self.__add_ones(self.g(Z))
        return A[1:, :].T

    def error(self, X, Y):
        return np.mean(np.square(self.predict(X) - Y), axis=0)
        
