import numpy as np
import torch
from src.plots import plot_errors

n_samples= 500
n_z=3
runs=5
M_choice="I"
m_max=8
path="../results/sum_of_sin/"
fig_path="sum_of_sin/"
# ---- Load data ----
e_as = torch.tensor(np.loadtxt(f"{path}/err_as_n{n_samples}_nadd{n_z}_{M_choice}.csv", delimiter=","))
e_mas = torch.tensor(np.loadtxt(f"{path}/err_mas_n{n_samples}_nadd{n_z}_{M_choice}.csv", delimiter=","))

plot_errors(e_as, e_mas, m_max, f"{fig_path}err_n{n_samples}_nadd{n_z}_{M_choice}.png")
