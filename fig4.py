import matplotlib.pyplot as plt
import numpy as np

# setting the x - coordinates
t = np.arange(0, 1, 0.001)
cases = [(1.0,1/6), (1.0,1/4)]
k = 0
for (a,b) in cases:
    J_1 = 25*(b**2)-t+t
    J_2 = a**2-t+t
    
    H_list = [(1,1), (0,1), (a**2,25*b**2)]
    for lamb in H_list:
        norm_grad = lamb[0]*(a**2)+25*(lamb[1])*(b**2)
        r = min(lamb)
        if r==0:
            r=1

        b_1 = 25*(b**2)*np.exp(-10*lamb[1]*t)+((1-np.exp(-2*r*t))/r)*norm_grad
        b_2 = (a**2)*np.exp(-2*lamb[0]*t)+((1-np.exp(-2*r*t))/r)*norm_grad

        b_min = np.minimum(b_1, np.minimum(J_1, J_2))
        # Plot
        fig, ax = plt.subplots(figsize=(6,4))

        ax.plot(t, b_1, color="#1f77b4", linewidth=2, label='MAS $U_m = U_{\{1\}}$')
        ax.plot(t, b_2, color="#ff7f0e", linewidth=2, label='MAS $U_m = U_{\{2\}}$')
        # Minimum curve
        ax.plot(t, J_1, color="#1f77b4", linestyle="dashed", linewidth=2, label='AS $U_m = U_{\{1\}}$')
        ax.plot(t, J_2, color="#ff7f0e", linestyle="dashed", linewidth=2, label='AS $U_m = U_{\{2\}}$')

        ax.plot(t, b_min, color="black", marker="*", markevery=40, linestyle='', 
                label='$\min$(AS,MAS)')

        # Axis labels with larger font
        ax.set_xlabel('t', fontsize=16)
        ax.set_ylabel('Error bound', fontsize=16)

        # Tick size
        ax.tick_params(axis='both', labelsize=14)

        ax.set_xticks(np.arange(0, 1, step=0.1))

        # Larger legend
        ax.legend(fontsize=14)

        plt.tight_layout()
        plt.savefig(f"figures/fig4_{k}.png")
        k+=1
        #plt.show()
