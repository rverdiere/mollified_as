import torch
import numpy as np
import itertools

class AS():
    def __init__(self, dim):
        self.dim=dim

    def matrix(self, x_grad):
        x_grad = x_grad.unsqueeze(-1)
        return (x_grad@x_grad.transpose(1,2)).mean(dim=0)
    def feature_map(self, x_grad, m):
        as_mat = self.matrix(x_grad)
        L,Q = torch.linalg.eigh(as_mat)
        bound = torch.sum(L[:self.dim-m])
        proj = Q[:, self.dim-m:]
        return bound, proj

