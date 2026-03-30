#################################################
#  Risk Decremental Behavior (eq. 9 - ICANN-2026)
#  by Marco Gori
#  30 March 2026
#################################################
import numpy as np
import matplotlib.pyplot as plt

# ==========================
# Parameters
# ==========================
L0 = 1          # Initial loss L(0)
alpha = 0.5       # Exponent (change this)
eta = 0.1       # Learning rate
Tmax = 50       # Maximum time to display
N = 10000       # Number of time points

# ==========================
# Time grid
# ==========================
t = np.linspace(0, Tmax, N)

# ==========================
# Compute theoretical loss
# ==========================
inside = L0**(1-alpha) - eta*(1-alpha)*t

# Terminal time (if alpha < 1)
if alpha < 1:
    T_terminal = L0**(1-alpha) / (eta*(1-alpha))
    print("Terminal time:", T_terminal)
else:
    T_terminal = None

# Avoid negative values after terminal time
L = np.maximum(inside, 0)**(1/(1-alpha))
# ==========================
# Plot
# ==========================
plt.figure()
plt.plot(t, L)
plt.xlabel("Time")
plt.ylabel("Loss")
plt.title(f"Theoretical Loss Decay (alpha={alpha})")
plt.yscale("linear")
plt.grid(True)
plt.show()
