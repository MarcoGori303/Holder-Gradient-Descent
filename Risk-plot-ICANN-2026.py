##########################################################
# Risk plot for singular, Holder, asymptotic behaviour
# ICANN-2026 risk plot
# by Marco Gori
# 30 March 2026
##########################################################
import numpy as np
import matplotlib.pyplot as plt

# --- Set Global Font to Times New Roman ---
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams['mathtext.fontset'] = 'stix' 

def calculate_L(t, L0, eta, alpha):
    if np.isclose(alpha, 1.0):
        return L0 * np.exp(-eta * t)
    else:
        exponent = 1.0 - alpha
        base_val = np.power(L0, exponent) - eta * exponent * t
        L = np.zeros_like(t, dtype=float)
        mask = base_val > 0
        L[mask] = np.power(base_val[mask], 1.0 / exponent)
        return L

# Parameters
L0 = 1.0
eta = 0.3
t = np.linspace(0, 15, 500)
alphas = [-0.5, 0.0, 0.2, 0.5, 0.667, 1.0, 1.5, 2.0]

plt.figure(figsize=(12, 7))

# Calculate boundary curves for the gray zone
L_05 = calculate_L(t, L0, eta, 0.5)
L_10 = calculate_L(t, L0, eta, 1.0)

# Fill the area between alpha=0.5 and alpha=1.0
plt.fill_between(t, L_05, L_10, color='gray', alpha=0.2)

# --- Add "Hölder regime" starting at x = 7 and very low ---
x_start = 7.0
y_close_to_axis = 0.02  # Keeps text sitting just above the x-axis

plt.text(x_start, y_close_to_axis, r'H$\ddot{\mathrm{o}}$lder regime', 
         fontsize=14, 
         color='black', 
         ha='left',   # Starts at x=7
         va='bottom', 
         rotation=0)

# Plotting Loop
for a in alphas:
    L_vals = calculate_L(t, L0, eta, a)
    if np.isclose(a, 0):
        thickness, z_order = 1, 5
    elif np.isclose(a, 0.5) or np.isclose(a, 1.0):
        thickness, z_order = 1, 4
    elif np.isclose(a, 0.667):
        thickness, z_order = 3, 4
    else:
        thickness, z_order = 1, 2

    plt.plot(t, L_vals, label=rf'$\alpha = {a}$', lw=thickness, zorder=z_order)

# Labels and Formatting
plt.title(r'Evolution of $e(t)$ for various $\alpha$', fontsize=22, pad=15)
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$e(t)$', fontsize=20)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=14, loc='upper right', frameon=True)
plt.grid(True, linestyle='--', alpha=0.4)
plt.ylim(0, L0 * 1.1)
plt.xlim(0, 15)

plt.tight_layout()
plt.show()
