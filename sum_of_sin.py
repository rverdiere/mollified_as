import matplotlib.pyplot as plt
import statistics as stats
import torch.nn as nn
import torch
import numpy as np
import itertools

from src.benchmark_functions import sum_of_sin
from torch.func import grad as fgrad
from src.mas import MAS
from src.active_sub import AS 
from src.train_featuremap import train_featuremap
from src.plots import plot_bounds

def tensor_to_csv(fname, x):
    np.savetxt(fname, x.numpy(), delimiter=",")

M_choice = "I"
train_f = False
d= 8
I_d = torch.eye(d)
m_int=6
a = torch.tensor([1,1/2, 1/3, 1/4,1/6, 1/8, 1/10,1/10])
omega = torch.tensor([1,4,5,6,7,9, 2,7])
u = sum_of_sin(a, omega)
n_add =3
n_samples = 500
n_samples_aug = (n_add+1)*n_samples
n_samples_test = 10000
n_run = 10
t_list = [0.1,0.01,0.003]
#t_list = [0.01]
m_max = d
path='results/sum_of_sin/'
fig_path='figures/sum_of_sin/'

bound_as = torch.zeros((n_run, m_max-1))
error_as = torch.zeros((n_run, m_max-1))
bound_mas = torch.zeros((n_run, m_max-1, len(t_list)))
error_mas = torch.zeros((n_run, m_max-1, len(t_list)))

x_test = torch.randn(n_samples_test,d)
y_test = u.func(x_test)

as_module= AS(d)
for run in range(n_run):
    x = torch.randn(n_samples,d)
    x_aug = torch.randn(n_samples_aug,d)
    y = u.func(x)
    y_aug = u.func(x_aug)[:n_samples]
    norm_y = torch.sqrt(torch.mean(y**2))
    
    x_grad = u.grad(x)
    x_grad_aug = u.grad(x_aug)

    if M_choice=="Hu":
        M=as_module.matrix(x_grad)
    elif M_choice=="Pi0":
        _,Q=as_module.feature_map(x_grad, m_int)
        M=Q@Q.t()
    else:
        M=I_d

    for j, t in enumerate(t_list):
        mas = MAS(t, M, n_add)
        x_add = mas.mollification_samples(x)
        x_grad_moll = mas.mollification_grad(x_add, u)
        for m in range(1,m_max):
            print(f"Run {run}, t={t}, m={m}")
            b_as, proj_as = as_module.feature_map(x_grad, m)
            bound_as[run, m-1]=b_as.item()
            b_mas, proj_mas = mas.feature_map(x_grad, x_grad_moll, m)            
            bound_mas[run, m-1, j]=b_mas.item()
            if train_f:
                if b_as<b_mas:
                    b_mas = b_as
                    proj_mas = proj_as
                for i, proj in enumerate((proj_as, proj_mas)):
                    f = nn.Sequential(nn.Linear(m, 20), nn.ReLU(), nn.Linear(20,20), nn.ReLU(), nn.Linear(20,1))
                    #if i==1:
                    #    _, f = train_featuremap(f, x_aug[n_samples:,], y, proj)
                    #else:
                    _, f = train_featuremap(f, x, y, proj)

                    x_test_red = x_test@proj
                    y_test_pred = f(x_test_red).squeeze()
                    err = ((y_test_pred-y_test)**2).mean()
                    if i==0:
                        error_as[run, m-1]=err.item()
                    else:
                        error_mas[run, m-1, j]=err.item()

if train_f:
    tensor_to_csv(f"{path}err_as_n{n_samples}_nadd{n_add}_{M_choice}.csv", error_as)
    tensor_to_csv(f"{path}err_mas_n{n_samples}_nadd{n_add}_{M_choice}.csv", error_mas.reshape((n_run, m_max-1)))
else:
    tensor_to_csv(f"{path}bound_as_n{n_samples}_nadd{n_add}_{M_choice}.csv", bound_as)
    for k,t in enumerate(t_list):
        tensor_to_csv(f"{path}bound_mas_n{n_samples}_nadd{n_add}_t{t}_{M_choice}.csv", bound_mas[:,:,k])
