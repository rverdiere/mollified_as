import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_errors(errors_as, errors_mas, m_max, fname):
    m = range(1, m_max)

    # ---- compute means ----
    e_as_mean = errors_as.mean(dim=0)
    #e_as_aug_mean = np.mean(e_as_aug, axis=0)
    e_mas_mean = errors_mas.mean(dim=0)

    fig, ax = plt.subplots(figsize=(6,4))
    # ---- individual runs ---
    for k in range(len(errors_as)):
        ax.semilogy(m, errors_as[k,:], ":", color='C0', linewidth=1, alpha=1)
        ax.semilogy(m, errors_mas[k,:], ":", color='C1', linewidth=1, alpha=1)

    # ---- mean curves ----
    ax.semilogy(m, e_as_mean, "-", color='C0', linewidth=3, label="AS")
    ax.semilogy(m, e_mas_mean, "-", color='C1', linewidth=3, label="MAS")

    ax.set_xlabel('$m$', fontsize=16)
    ax.set_ylabel('Approximation error', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    ax.legend(frameon=False, fontsize=14)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()

    plt.savefig(fname, dpi=300)

    plt.show()

def plot_bounds(bound_as, bound_mas, t_list, m_max, fname):
    x = torch.tensor(list(range(1,m_max)))
    b_as_mean =  bound_as.mean(dim=0)
    b_as_std =  bound_as.std( dim=0)
    b_mas_mean =  bound_mas.mean(dim=0)
    b_mas_std =  bound_mas.std( dim=0)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.semilogy(x, b_as_mean, "-", color='C0', linewidth=3, label='AS')
    ax.fill_between(
        x,
        b_as_mean - b_as_std,
        b_as_mean + b_as_std,
        color='C0',
        alpha=0.15
    )

    # ---- MAS for different t ----
    for k, t_val in enumerate(t_list):
        ax.semilogy(
            x,
            b_mas_mean[:, k],
            "-",
            linewidth=2.5,
            label=f"MAS (t={t_val})",
            color='C'+str(k+1)
        )
        
        ax.fill_between(
            x,
            b_mas_mean[:, k] - b_mas_std[:, k],
            b_mas_mean[:, k] + b_mas_std[:, k],
            alpha=0.15,
            color='C'+str(k+1)
        )

    # ---- formatting ----
    ax.set_xlabel('$m$', fontsize=16)
    ax.set_ylabel('Error bound', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    ax.legend(frameon=False, fontsize=14)

    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    plt.tight_layout()

    plt.savefig(fname, dpi=300)
    plt.show()
