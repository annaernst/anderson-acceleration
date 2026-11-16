import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh



n = 500

rng = np.random.default_rng(0)
b = rng.random(n)

eigenvals = np.linalg.eigvalsh()

lambda_min = eigenvals[0]
lambda_max = eigenvals[-1]

alpha = 2 / (lambda_min + lambda_max)

def Richardson_iteration(A, b,alpha):
    x = np.zeros(n)
    errors = []

    a = alpha

    for i in range(1,500):
        x_new = x - (a * (A @ x - b))
        e_i = np.linalg.norm((A @ x_new) - b)
        errors.append(e_i)

        if e_i < 0.00000001:
            return x_new, errors
        x = x_new
    
    return x, errors


R_output, R_error = Richardson_iteration(A,b, alpha)

plt.semilogy(R_error, label="Richardson Iteration")
plt.xlabel("sweeps k")
plt.ylabel(r"error $||Ax-b||$")
plt.title("Convergence of Richardson Iteration")
plt.legend()
plt.show()


print(f"Average outcome for Richardson: {R_output[9]:.9f}")