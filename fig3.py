import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.hermite_e import hermeval
from math import factorial, sqrt

# Degrees we want
degrees = [1, 5, 20]

# X values for plotting
x = np.linspace(-2, 2, 400)

fig, ax = plt.subplots(figsize=(8,6))

for n in degrees:
    # Coefficients for He_n(x)
    coeffs = [0] * n + [1]
    He_n = hermeval(x, coeffs)
    
    # L^2 normalization: divide by sqrt(n!)
    psi_n = He_n / sqrt(factorial(n))
    
    ax.plot(x, psi_n, label=f"$\psi_{{{n}}}(x)$", linewidth=2)

plt.title("$L^2$-normalized Hermite Polynomials", fontsize=14)
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel("$\psi_n(x)$", fontsize=16)
ax.grid(True)
ax.legend(fontsize=14)
plt.tight_layout()
plt.savefig("figures/fig3.png")
plt.show()

