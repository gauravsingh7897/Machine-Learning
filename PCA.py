import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.X = None
        self.eigen_vec = None

    def fit(self, X):
        self.X = X
        X = self.standardized()
        
        covarinace = np.cov(X.T)
        eigen_val, eigen_vec = np.linalg.eig(covarinace)
        self.eigen_vec = eigen_vec

    def transform(self):
        if self.X is None:
            return "Please fit PCA."
        X = self.X
        Y = np.matmul(self.eigen_vec, X.T).T
        return Y[:,:self.n_components]
    
    def standardized(self):
        if self.X is None:
            return
        mean = np.mean(self.X, axis=0)
        stddev = np.std(self.X, axis=0)

        self.X = (self.X - mean)/ stddev
        
        return self.X


x = np.random.normal(size=(200,2))

pca = PCA(n_components=1)
pca.fit(x)
y = pca.transform()
print(y.shape)