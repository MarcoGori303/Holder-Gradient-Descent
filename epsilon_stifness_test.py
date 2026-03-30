################################################################
# Holder Gradient Descent
# Explicit method with different values of epsilon
# by Marco Gori
# 30 March 2026
################################################################
import numpy as np
import matplotlib.pyplot as plt

# XOR Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_der(x): return x * (1 - x)

def get_grads(w1, w2):
    # Forward Pass
    a1 = sigmoid(np.dot(X, w1))
    a2 = sigmoid(np.dot(a1, w2))
    error = a2 - y
    loss = np.mean(0.5 * (error**2))
    
    # Backward Pass
    dw2 = np.dot(a1.T, error * sigmoid_der(a2))
    delta_hidden = np.dot(error * sigmoid_der(a2), w2.T) * sigmoid_der(a1)
    dw1 = np.dot(X.T, delta_hidden)
    return loss, dw1, dw2

def train_explicit(alpha, epsilon, lr=0.05, epochs=1000):
    np.random.seed(42)
    w1 = np.random.uniform(-1, 1, (2, 4))
    w2 = np.random.uniform(-1, 1, (4, 1))
    history = []
    
    for _ in range(epochs):
        loss, dw1, dw2 = get_grads(w1, w2)
        history.append(loss)
        
        # Calculate squared norm of the total gradient vector
        g_flat = np.concatenate([dw1.flatten(), dw2.flatten()])
        gn2 = np.sum(g_flat**2)
        
        # The Explicit Update: w_dot = -lr * (E^alpha / (||grad||^2 + eps)) * grad
        # If epsilon is 0 and gn2 is very small, 'mag' becomes huge.
        mag = lr * (loss**alpha) / (gn2 + epsilon)
        
        w1 -= mag * dw1
        w2 -= mag * dw2
        
        # Stability Check: if weights blow up or loss becomes NaN, stop.
        if np.isnan(loss) or loss > 1e2:
            return history + [np.nan] * (epochs - len(history))
            
    return history

# Parameters
alphas = [0.5, 0.66, 1.0]
epsilons = [1e-1, 1e-4, 1e-6, 1e-8] # High, Medium, and Zero damping

fig, axes = plt.subplots(len(epsilons), 1, figsize=(10, 12), sharex=True)

for i, eps in enumerate(epsilons):
    ax = axes[i]
    for a in alphas:
        hist = train_explicit(a, eps)
        ax.plot(hist, label=f'α={a:.2f}')
    
    ax.set_yscale('log')
    ax.set_title(f'Explicit Euler Stability: ε = {eps}')
    ax.set_ylabel('Loss (Log)')
    ax.grid(True, alpha=0.3)
    if i == 0: ax.legend()

plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
