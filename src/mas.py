import torch
import numpy as np
import itertools
from torch import vmap

def subsets(n ,d):
    lst = []
    item = [k for k in range(d)]
    for l in range(1, n+1):
        for subset in itertools.combinations(item, l):
            lst.append(list(subset))
    return lst

def one_matrices(pos_list, d):
    ret = torch.zeros(len(pos_list), d, d)
    k=0
    for lst in pos_list:
        ret[k, :, :]=torch.diag(torch.sum(torch.nn.functional.one_hot(torch.tensor(lst), d), dim=0))
        k+=1
    return ret

def is_identity(A, atol=1e-8, rtol=1e-5):
    return torch.allclose(A,
        torch.eye(A.shape[-1]),
        atol=atol,
        rtol=rtol)

class MAS():
    def __init__(self, t, M, n_add):
        self.t = t
        self.M = M
        self.n_add = n_add
        self.dim = M.size()[0]
        if is_identity(M):
            self.commut = False
        else:
            self.commut = True
     
    def projector(self, m, Q):
        id_list = subsets(m, self.dim)
        proj_list = one_matrices(id_list, self.dim)
        return Q@proj_list@Q.t()


    def matrix(self, x, y):
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        A = torch.linalg.matrix_exp(-self.t*self.M)
        B = (x@(y.transpose(1,2))).mean(dim=0)
        B_sym = (B+B.t())/2
        return A@B_sym@A
    
    def grad_norm(self, x_grad):
        n, d = x_grad.size()
        x_grad = x_grad.unsqueeze(-1)
        return (x_grad.transpose(1,2)@self.M@x_grad).mean(dim=0)

    def feature_map(self, x_grad, y_grad, m):
        mas_mat = self.matrix(x_grad,y_grad)
        L,Q = torch.linalg.eigh(mas_mat)
        L[L<0]=0
        grad_norm = self.grad_norm(x_grad)
        L_M,Q_M = torch.linalg.eigh(self.M)
        r=torch.min(L_M[L_M>0])
        I_d = torch.eye(self.dim)
        if not self.commut:
            bound = torch.sum(L[:self.dim-m])+((1-np.exp(-2*r*self.t))/r)*grad_norm
            proj = Q[:,self.dim-m:]
        else:
            pi = self.projector(m,Q)
            res = vmap(torch.trace)((I_d-pi)@mas_mat)
            bound = torch.min(res)+((1-np.exp(-2*r*self.t))/r)*grad_norm
            L_pip, Q_pi = torch.linalg.eigh(pi[torch.argmin(res), :, :])
            proj = Q_pi[:, self.dim-m:]
        
        return bound, proj

    def mollification_samples(self, x):
        """
        Generate n_z new samples for each input point X.

        Parameters
        ----------
        x : torch.Tensor
            Shape (batch_size, d).

        Returns
        -------
        x_molli : torch.Tensor
            Shape (batch_size, n_z, d).
        """
        n = x.size()[0]
        z = torch.randn(n, self.n_add, self.dim)

        A = torch.matrix_exp(-2 * self.t * self.M)

        identity = torch.eye(self.dim)
        B = identity - torch.matrix_exp(-4 * self.t * self.M)

        # Symmetric matrix square root of B
        L, Q = torch.linalg.eigh(B)
        L = L.clamp_min(0)
        B_sqrt = Q@torch.diag(torch.sqrt(L))@Q.transpose(-1, -2)

        Ax = x @ A.transpose(-1, -2)
        Bz = z @ B_sqrt.transpose(-1, -2)
        x_molli =Ax.unsqueeze(1) + Bz

        return x_molli
    
    def mollification_grad(self, x_add, u):
        n, n_z, d = x_add.size()  
        x_add_grad = u.grad(x_add.view(-1, d)).reshape(n, n_z, d)
        return x_add_grad.mean(dim=1)

