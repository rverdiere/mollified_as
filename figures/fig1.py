import numpy as np
import matplotlib.pyplot as plt

# Time vector
t = np.arange(0, 0.05, 0.0001)

# Define curves
b_1 = 100/72 * np.exp(-100*(1 - np.exp(-2*t)) - 2*t) * (1 + np.exp(-200*np.exp(-2*t)))
b_2 = 1/2 * np.exp(-1*(1 - np.exp(-2*t)) - 2*t) * (1 + np.exp(-2*np.exp(-2*t)))
e = 0.14 + t - t  # constant line at 0.14
e2 = 0.57 + t - t  # constant line at 0.57

# Find the first intersection point between b_1 and b_2
idx_cross = np.where(np.diff(np.sign(b_1 - b_2)))[0][0]
t_cross = t[idx_cross]
y_cross = b_1[idx_cross]

# Plot
fig, ax = plt.subplots(figsize=(6,4))

ax.plot(t, b_1, color="C0", label='$P_t(u)$ AS bound for $U_m=U_{1}$')
ax.plot(t, b_2, color="C1", label='$P_t(u)$ AS bound for $U_m=U_{2}$')
ax.plot(t, e, color="C0", linestyle='--', label='$u$ reconstruction error for $U_m=U_{1}$')
ax.plot(t, e2, color="C1", linestyle='--', label='$u$ reconstruction error for $U_m=U_{2}$')

# Vertical line at intersection
ax.axvline(t_cross, color='gray', linestyle=':', linewidth=1)

ax.annotate(f'${t_cross:.3f}$',
            xy=(t_cross, 0), xytext=(t_cross, -0.092),
            ha='center', va='top',
            fontsize=10)

# Axes and labels (larger)
ax.set_xlabel('$t$', fontsize=16)

# Tick size
ax.tick_params(axis='both', labelsize=14)

ax.set_xticks(np.arange(0, 0.05, step=0.01))
ax.set_ylim(bottom=-0.05)

# Larger legend
ax.legend(fontsize=14)

plt.tight_layout()
plt.savefig("fig1.png")
plt.show()
