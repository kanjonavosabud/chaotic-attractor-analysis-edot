import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Lorenz system parameters
sigma = 10
beta = 8/3
rho = 22  # Given in your problem

# Compute equilibrium point (assuming non-trivial one: x = y = sqrt(beta(rho-1)), z = rho-1)
x_eq = np.sqrt(beta * (rho - 1))
y_eq = np.sqrt(beta * (rho - 1))
z_eq = rho - 1
 
# Compute the Jacobian matrix at the equilibrium point
A = np.array([[-sigma, sigma, 0],
              [1, -1, -x_eq],
              [y_eq, x_eq, -beta]])

# Define B, C, and D for state-space representation
B = np.array([[1], [0], [0]])  # Assume input perturbation on x
C = np.array([[1, 0, 0]])  # Observe only x (you can change this)
D = np.array([[0]])  # No direct feedthrough

# Create state-space system
system = signal.StateSpace(A, B, C, D)

# Compute frequency response for Nyquist plot
w, H = signal.freqresp(system)

# Nyquist Plot
plt.figure(figsize=(6, 6))
plt.plot(H.real, H.imag, label="Nyquist Plot")
plt.plot(H.real, -H.imag, linestyle='dashed', color='gray', alpha=0.7)  # Mirror image

plt.scatter([-1], [0], color="red", marker="x", label="Critical Point (-1,0)")
plt.xlabel("Re")
plt.ylabel("Im")
plt.title("Nyquist Plot (Linearized Lorenz System at E_+)")
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid()
plt.show()