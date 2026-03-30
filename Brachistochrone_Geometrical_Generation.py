import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_cycloid(u0=10):
    R = u0 / 2
    # Parameter eta from 0 to pi (half an arch)
    eta_vals = np.linspace(0, np.pi, 200)
    
    # Pre-calculate the cycloid path
    x_path = R * (eta_vals - np.sin(eta_vals))
    y_path = R * (1 - np.cos(eta_vals))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, R * np.pi + 1)
    ax.set_ylim(-u0 - 1, 1)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f"Rolling Circle Generation (R = {R})")
    ax.set_xlabel("Horizontal Distance (x)")
    ax.set_ylabel("Vertical Drop (-y)")

    # Plot the 'ceiling'
    ax.axhline(0, color='black', lw=2)
    
    # Elements to animate
    path_line, = ax.plot([], [], 'r-', lw=2, label="Cycloid Path")
    circle = plt.Circle((0, -R), R, fill=False, color='blue', lw=2, label="Generating Circle")
    point, = ax.plot([], [], 'ko') # The tracing point
    radius_line, = ax.plot([], [], 'b--', lw=1) # Connects center to point
    
    ax.add_patch(circle)
    ax.legend()

    def update(eta):
        # Center of circle moves along x = R * eta
        cx = R * eta
        cy = -R
        circle.center = (cx, cy)
        
        # Point on the rim
        px = R * (eta - np.sin(eta))
        py = -R * (1 - np.cos(eta))
        
        path_idx = np.where(eta_vals <= eta)[0]
        path_line.set_data(x_path[path_idx], -y_path[path_idx])
        point.set_data([px], [py])
        radius_line.set_data([cx, px], [cy, py])
        
        return path_line, circle, point, radius_line

    ani = FuncAnimation(fig, update, frames=eta_vals, blit=True, interval=20)
    plt.show()

animate_cycloid(u0=10)
