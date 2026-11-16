import numpy as np
from scipy.io import mmread
from scipy.sparse import eye
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt

def richardson_anderson(A, b, alpha, m=0, tol=1e-8, max_iter=10000):
    #m=0 anderson acceleration is regular richardson
    x = np.zeros_like(b)
    x_list = []
    f_list = []
    g_list = []

    for k in range(max_iter):
        x_list.append(x)
        r = A @ x - b
        if np.linalg.norm(r) / np.linalg.norm(b) < tol:
            return x_list
        g = x - alpha * r #Richardson
        f_list.append(g - x)
        g_list.append(g)
        m_k = min(m, k)
        if m_k > 0:
            F = np.column_stack([
                f_list[i+1] - f_list[i]
                for i in range(len(f_list)-m_k-1, len(f_list)-1)
            ])
            G = np.column_stack([
                g_list[i+1] - g_list[i]
                for i in range(len(g_list)-m_k-1, len(g_list)-1)
            ])

            gamma = np.linalg.lstsq(F, f_list[-1], rcond=None)[0]
            x = g - G @ gamma
        else:
            x = g

        f_list = f_list[-(m+1):]
        g_list = g_list[-(m+1):]

    #only if didn't converge
    x_list.append(x)
    return x_list

A = mmread("gr_30_30.mtx").tocsr() # compressed sparse row format: data, indices, index pointer


# Compute optimal alpha for a given A
lambda_min = eigs(A, k=1, which="SR", return_eigenvectors=False)[0].real
lambda_max = eigs(A, k=1, which="LR", return_eigenvectors=False)[0].real
alpha = 2 / (lambda_min + lambda_max)

# Calculate spectral radius of B := I-alpha*A, need < 1 to guarantee convergence
B = eye(A.shape[0]) - alpha * A
spec_rad = abs(eigs(B, k=1, which="LM", return_eigenvectors=False)[0])
print("Spectral radius:", spec_rad)

# Solve Ax = b
rng = np.random.default_rng(42)
x_exact = rng.random(A.shape[0])
b = A @ x_exact


for m in range(7):
    x_list = richardson_anderson(A, b, alpha, m)
    print("Iterations with m = ", m, ":", len(x_list) - 1)
    errors = [
        np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact)
        for x in x_list
    ]
    if m == 0:
        plt.semilogy(errors, label=f"m={m}", linestyle="--")
    else:
        plt.semilogy(errors, label=f"m={m}")


plt.xlabel("Iteration")
plt.ylabel("Relative solution error")
plt.title("Richardson Iteration with Anderson Acceleration")
plt.grid(True)
plt.legend()
plt.show()