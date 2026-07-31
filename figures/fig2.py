import matplotlib.pyplot as plt
import numpy as np

# setting the x - coordinates
sig=0.2
t=0.05
x = np.arange(-10, 10, 0.01)
# setting the corresponding y - coordinates
y1=np.sin(x)+1/6*np.sin(10*x)
y2=np.sin(np.exp(-t)*x)*np.exp(-(1-np.exp(-2*t))/2)+np.sin(10*np.exp(-t)*x)*np.exp(-100*(1-np.exp(-2*t))/2)

# plotting the points
fig, ax = plt.subplots(figsize=(6,4))
ax.plot(x, y1, label='$u$', linewidth=2)
ax.set_xlabel('x', fontsize=16)
ax.plot(x, y2, label='$P_{t}(u)$', linewidth=2)

ax.tick_params(axis='both', labelsize=14)
# function to show the plot
ax.legend(fontsize=14, loc='upper left')
plt.tight_layout()
plt.savefig("fig3_t"+str(t)+".png")
plt.show()
