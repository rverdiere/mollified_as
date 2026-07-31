import numpy as np
import torch
from src.plots import plot_errors, plot_bounds

n_samples= 300
n_samples_aug= 3000
n_z=9
runs=2
m_max=8
t_list=[0.001]
path="results/wave_model/"
fig_path="figures/wave_model/"
#Bounds
bound_as = torch.tensor(np.loadtxt(f"{path}bound_as_n{n_samples}.csv", delimiter=","))
bound_as_aug = torch.tensor(np.loadtxt(f"{path}bound_as_aug_n{n_samples_aug}.csv", delimiter=","))
bound_mas = torch.tensor(np.loadtxt(f"{path}bound_mas_n{n_samples}_nadd{n_z}.csv", delimiter=","))
print(bound_mas)
plot_bounds(bound_as, bound_mas.unsqueeze(-1), t_list, m_max, f"{fig_path}bound_n{n_samples}_nadd{n_z}.png")
plot_bounds(bound_as_aug, bound_mas.unsqueeze(-1), t_list, m_max, f"{fig_path}bound_aug_n{n_samples}_nadd{n_z}.png")

# ---- Load data ----
e_as = torch.tensor(np.loadtxt(f"{path}err_as_n{n_samples}.csv", delimiter=","))
e_as_aug = torch.tensor(np.loadtxt(f"{path}err_as_aug_n{n_samples_aug}.csv", delimiter=","))
e_mas = torch.tensor(np.loadtxt(f"{path}err_mas_n{n_samples}_nadd{n_z}.csv", delimiter=","))

plot_errors(e_as, e_mas, m_max, f"{fig_path}err_n{n_samples}_nadd{n_z}.png")
plot_errors(e_as_aug, e_mas, m_max, f"{fig_path}err_aug_n{n_samples}_nadd{n_z}.png")
