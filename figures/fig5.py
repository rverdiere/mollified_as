import numpy as np
import torch
from src.plots import plot_bounds

n_samples= 500
n_z=3
M_choice="I"
t_list = [0.1,0.01,0.003]
path="../results/sum_of_sin/"
fig_path="sum_of_sin/"
# ---- Load data ----
bound_as = torch.tensor(np.loadtxt(f"{path}/bound_as_n{n_samples}_nadd{n_z}_{M_choice}.csv", delimiter=","))
runs, m_max = bound_as.size()
bound_mas = torch.zeros((runs, m_max, len(t_list)))
for k,t in enumerate(t_list):
    bound_mas[:,:,k] = torch.tensor(np.loadtxt(f"{path}/bound_mas_n{n_samples}_nadd{n_z}_t{t}_{M_choice}.csv", delimiter=","))

plot_bounds(bound_as, bound_mas, t_list, m_max+1, f"{fig_path}bound_n{n_samples}_nadd{n_z}_{M_choice}.png")
