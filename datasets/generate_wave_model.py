from fenics import *
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from functools import partial
from time import time
from scipy.linalg import expm

# -----------------------------
# Simulation parameters
# -----------------------------
nx = 100
num_steps = 200
T = 1.0
dt = T / num_steps
K = 8
x0 = 0.7
h = 1e-3
#control parameters
initial_samples=True
MAS=True
test=False
n_samples_test = 1000
n_samples_train = 300
runs=10
#MAS
n_z=9
t=0.001
M = np.eye(K)

# -----------------------------
# Precomputations
# -----------------------------
mesh = IntervalMesh(nx, 0.0, 1.0)
V = FunctionSpace(mesh, "CG", 1)
x_vals = mesh.coordinates().flatten()
probe_index = np.argmin(np.abs(x_vals - x0))

u0_expr = Expression("exp(-100*pow(x[0] - 0.5, 2))", degree=2)
u_init = interpolate(u0_expr, V)

u_trial = TrialFunction(V)
v_test = TestFunction(V)
bc = DirichletBC(V, Constant(0.0), "on_boundary")

# -----------------------------
# UserExpression for c(x)
# -----------------------------
class ParametricC(UserExpression):
    def __init__(self, theta, **kwargs):
        self.theta = theta
        self.K = len(theta)
        super().__init__(**kwargs)

    def eval(self, values, x):
        a = self.theta[:int(self.K/2)]
        omega = self.theta[int(self.K/2):]
        val = 1.0
        for k in range(int(self.K/2)):
            val += (1/(2**(k+1)))*(a[k]-1/2)*np.sin(10**k*omega[k]*x[0])
        #val += 1/2*(theta[6]-1/2)*np.sin(10*self.theta[1]*x[0])
        #val += 1/4*(theta[7]-1/2)*np.sin(100*self.theta[2]*x[0])
        #val += 1/8*(theta[8]-1/2)*np.sin(500*self.theta[3]*x[0])
        #val += 1/16*(theta[8]-1/2)*np.sin(1000*self.theta[4]*x[0])
        #for k in range(self.K):
        #    val += 2*self.theta[k] * np.sin(2 * np.pi * (k + 1) * x[0])

        values[0] = val

    def value_shape(self):
        return ()

# -----------------------------
# Solver
# -----------------------------
def solve_wave(theta):
    #Transport parameters theta
    theta = scipy.stats.norm.cdf(theta)
    
    c_expr = ParametricC(theta, degree=5)
    c = interpolate(c_expr, V)

    # Initial conditions
    u_n = u_init.copy(deepcopy=True)
    u_nm1 = u_init.copy(deepcopy=True)
    u = Function(V)

    a = u_trial * v_test * dx + dt**2 * dot(c * c * grad(u_trial), grad(v_test)) * dx
    A = assemble(a)
    bc.apply(A)

    for _ in range(num_steps):
        L = 2 * u_n * v_test * dx - u_nm1 * v_test * dx
        b = assemble(L)
        bc.apply(b)
        solve(A, u.vector(), b)
        u_nm1.assign(u_n)
        u_n.assign(u)

    return u.vector()[probe_index]

# -----------------------------
# Gradient Estimation Helper
# -----------------------------
def estimate_gradient(theta):
    grad = np.zeros(K)
    for k in range(K):
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[k] += h
        theta_minus[k] -= h
        f_plus = solve_wave(theta_plus)
        f_minus = solve_wave(theta_minus)
        grad[k] = (f_plus - f_minus) / (2 * h)
    return grad

def worker_gradient(theta_sample):
    return estimate_gradient(theta_sample)

def estimate_gradient_para(theta_samples):
    print("Estimating gradients in parallel...")
    start_time = time()
    with Pool(processes=cpu_count()) as pool:
        Grads = pool.map(worker_gradient, theta_samples)
    Grads = np.array(Grads)
    print("Gradient estimation done in {:.2f}s".format(time() - start_time))
    return Grads

def compute_new_samples(X, t, Z, M):
    d = M.shape[0]
    n = Z.shape[0]
    A  = expm(-2*t*M)
    B  = np.eye(d)-expm(-4*t*M)
    L_B,Q_B = np.linalg.eigh(B)
    L_B[L_B<0]=0.
    B_sqrt = (Q_B@np.sqrt(np.diag(L_B))@Q_B.T).reshape(1,d,d)
    Z = Z.reshape(n, d, 1)
    Y_adj = A@X+(B_sqrt@Z).reshape(n,d)
    return Y_adj


# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    if test==True:
        Theta = np.random.normal(size=(n_samples_test, K))
        QoIs = np.zeros(n_samples_test)
        print("Test mode")
        print("Computing QoIs...")
        start_time = time()
        for i in range(n_samples_test):
            print(f"Sample {i+1}/{n_samples_test}", end="\r")
            QoIs[i] = solve_wave(Theta[i])
        print("QoI computation done in {:.2f}s".format(time() - start_time))

        pd.DataFrame(Theta).to_csv("data_wave/parameters"+str(n_samples_test)+"_test.csv", header=False, index=False)
        pd.DataFrame(QoIs).to_csv("data_wave/qois"+str(n_samples_test)+"_test.csv", header=False, index=False)
    else:
        for r in range(2, runs):
            print(f"Run {r+1}/{runs}")
            Theta = np.random.normal(size=(n_samples_train, K))
            QoIs = np.zeros(n_samples_train)
            print("Computing QoIs...")
            start_time = time()
            for i in range(n_samples_train):
                print(f"Sample {i+1}/{n_samples_train}", end="\r")
                QoIs[i] = solve_wave(Theta[i])
            print("QoI computation done in {:.2f}s".format(time() - start_time))

            pd.DataFrame(Theta).to_csv("data_wave/parameters"+str(n_samples_train)+"_"+str(r)+"_t"+str(t)+".csv", header=False, index=False)
            pd.DataFrame(QoIs).to_csv("data_wave/qois"+str(n_samples_train)+"_"+str(r)+"_t"+str(t)+".csv", header=False, index=False)
            
            print("Estimating gradients in parallel...")
            start_time = time()
            with Pool(processes=cpu_count()) as pool:
                Grads = pool.map(worker_gradient, Theta)
            Grads = np.array(Grads)
            print("Gradient estimation done in {:.2f}s".format(time() - start_time))
            pd.DataFrame(Grads).to_csv("data_wave/grads"+str(n_samples_train)+"_"+str(r)+"_t"+str(t)+".csv", header=False, index=False)
            if MAS:
                if not initial_samples:
                    Theta = pd.read_csv('data_wave/parameters'+str(n_samples_train)+'_'+str(r)+"_t"+str(t)+'.csv', sep=',', header=None).values
                print("MAS")
                Z = np.random.randn(n_z, K)
                X_aug = np.zeros((n_samples_train, n_z, K))
                print(Theta.shape)
                for k in range(n_samples_train):
                    X_aug[k] = compute_new_samples(Theta[k],t,Z,M)
                print(X_aug.shape)
                X_aug = X_aug.reshape(n_samples_train*n_z,K)

                X_aug_grad = estimate_gradient_para(X_aug)
                pd.DataFrame(X_aug).to_csv("data_wave/smoothing_sample"+str(n_samples_train)+"_"+str(r)+"_t"+str(t)+".csv", header=False, index=False)
                pd.DataFrame(X_aug_grad).to_csv("data_wave/grad_add"+str(n_samples_train)+"_"+str(r)+"_t"+str(t)+".csv", header=False, index=False)

