import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from src.mas import MAS
from src.active_sub import AS
from src.plots import plot_bounds
from src.train_featuremap import train_featuremap

def tensor_to_csv(fname, x):
    np.savetxt(fname, x.numpy(), delimiter=",")

#Parameters
dset_path = "datasets/wave_model/"
res_path = "results/wave_model/"
fig_path = "figures/wave_model/"
n_train = 300
n_train_aug = 3000
n_test= 1000
d = 8
n_add=9
t=0.001
n_run=2
m_max=d
M = torch.eye(d)
train_f=False
max_epoch= 300
print_freq=150

# AS module
as_module= AS(d)

#Bound and error tensors
bound_as = torch.zeros((n_run, m_max-1))
error_as = torch.zeros((n_run, m_max-1))
bound_as_aug = torch.zeros((n_run, m_max-1))
error_as_aug = torch.zeros((n_run, m_max-1))
bound_mas = torch.zeros((n_run, m_max-1))
error_mas = torch.zeros((n_run, m_max-1))

# Loading testing set
x_test = torch.tensor(np.loadtxt(f"{dset_path}parameters{n_test}_test.csv", delimiter=','), dtype=torch.float32)
y_test = torch.tensor(np.loadtxt(f"{dset_path}qois{n_test}_test.csv", delimiter=','), dtype=torch.float32)

for r in range(n_run):
    x = torch.tensor(np.loadtxt(f"{dset_path}parameters{n_train}_{r}_t{t}.csv", delimiter=','), requires_grad=True, dtype=torch.float32)
    y = torch.tensor(np.loadtxt(f"{dset_path}qois{n_train}_{r}_t{t}.csv", delimiter=','), dtype=torch.float32)
    x_grad = torch.tensor(np.loadtxt(f"{dset_path}grads{n_train}_{r}_t{t}.csv", delimiter=','), dtype=torch.float32)


    x_aug = torch.tensor(np.loadtxt(f"{dset_path}parameters{n_train_aug}_{r}.csv", delimiter=','), requires_grad=True, dtype=torch.float32)
    y_aug = torch.tensor(np.loadtxt(f"{dset_path}qois{n_train_aug}_{r}.csv", delimiter=','), dtype=torch.float32)
    x_grad_aug = torch.tensor(np.loadtxt(f"{dset_path}grads{n_train_aug}_{r}.csv", delimiter=','), dtype=torch.float32)

    x_grad_moll = torch.tensor(np.loadtxt(f"{dset_path}grad_add{n_train}_{r}_t{t}.csv", delimiter=','), dtype=torch.float32)
    x_moll = torch.tensor(np.loadtxt(f"{dset_path}smoothing_sample{n_train}_{r}_t{t}.csv", delimiter=','), dtype=torch.float32)
    x_grad_moll = x_grad_moll.reshape((n_train, n_add, d)).mean(dim=1)
    mas = MAS(t, M, n_add)
    for m in range(1,m_max):
        print(f"Run {r}, t={t}, m={m}")
        b_as, proj_as = as_module.feature_map(x_grad, m)
        bound_as[r, m-1]=b_as.item()
        b_as_aug, proj_as_aug = as_module.feature_map(x_grad_aug, m)
        bound_as_aug[r, m-1]=b_as_aug.item()
        b_mas, proj_mas = mas.feature_map(x_grad, x_grad_moll, m)            
        bound_mas[r, m-1]=b_mas.item()
        if train_f:
            if b_as<b_mas:
                #b_mas = b_as
                proj_mas = proj_as
            for i, proj in enumerate((proj_as, proj_as_aug, proj_mas)):
                f = nn.Sequential(nn.Linear(m, 20), nn.ReLU(), nn.Linear(20,20), nn.ReLU(), nn.Linear(20,1))
                if i==1:
                    _, f = train_featuremap(f, x_aug[:n_train,], y_aug[:n_train], proj, max_epoch=max_epoch, print_freq=print_freq)
                else:
                    _, f = train_featuremap(f, x, y, proj,max_epoch=max_epoch, print_freq=print_freq)

                x_test_red = x_test@proj
                y_test_pred = f(x_test_red).squeeze()
                err = ((y_test_pred-y_test)**2).mean()
                if i==0:
                    error_as[r, m-1]=err.item()
                elif i==1:
                    error_as_aug[r, m-1]=err.item()
                else:
                    error_mas[r, m-1]=err.item()

if train_f:
    tensor_to_csv(f"{res_path}err_as_n{n_train}.csv", error_as)
    tensor_to_csv(f"{res_path}err_as_aug_n{n_train_aug}.csv", error_as_aug)
    tensor_to_csv(f"{res_path}err_mas_n{n_train}_nadd{n_add}.csv", error_mas.reshape((n_run, m_max-1)))
    #plot_errors(error_as, error_mas, m_max, f"{fig_path}/err_n{n_samples}_nadd{n_add}_{M_choice}.png")
else:
    tensor_to_csv(f"{res_path}bound_as_n{n_train}.csv", bound_as)
    tensor_to_csv(f"{res_path}bound_as_aug_n{n_train_aug}.csv", bound_as_aug)
    tensor_to_csv(f"{res_path}bound_mas_n{n_train}_nadd{n_add}.csv", bound_mas.reshape((n_run, m_max-1)))
    #plot_bounds(bound_as, bound_mas.unsqueeze(-1), [t], m_max, f"{fig_path}/bound_wave_n{n_train}_t{t}.png")
    #plot_bounds(bound_as_aug, bound_mas.unsqueeze(-1), [t], m_max, f"{fig_path}/bound_wave_aug_n{n_train_aug}_t{t}.png")

