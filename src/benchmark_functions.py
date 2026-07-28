import torch
from torch import vmap
from torch.func import grad

class sum_of_sin:
    def __init__(self, a, omega):
        self.name="sum_of_sin"
        self.dim=len(a)
        self.a=a
        self.omega=omega
    
    def func(self, X):
        if X.dim()==1:
            return torch.sum(self.a*torch.sin(self.omega*X))
        else:
            omega = self.omega.reshape((1,self.dim))
            a = self.a.reshape((1,self.dim))
            return (a*torch.sin(omega*X)).sum(axis=1)

    def grad(self, X):
        if X.dim()==1:
            return grad(self.func)(X)
        else:
            return vmap(grad(self.func))(X)
