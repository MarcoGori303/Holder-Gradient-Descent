####################################################################
# Holder method with alpha = 2/3 - optimal solution
# epsilon_explicit vs implicit methods
# XOR neural network
# by Marco Gori, 30 March 2026
####################################################################
import numpy as np
import matplotlib.pyplot as plt

# XOR Data
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
def sigmoid_der(x): return x * (1 - x)

def get_grads(w1, w2):
    a1 = sigmoid(np.dot(X, w1))
    a2 = sigmoid(np.dot(a1, w2))
    error = a2 - y
    loss = np.mean(0.5 * (error**2))
    dw2 = np.dot(a1.T, error * sigmoid_der(a2))
    delta_hidden = np.dot(error * sigmoid_der(a2), w2.T) * sigmoid_der(a1)
    dw1 = np.dot(X.T, delta_hidden)
    return loss, dw1, dw2

def train(alpha, mode, lr, eps_val=1e-6, epochs=2000):
    np.random.seed(42) # Ensure same starting point for fair comparison
    w1 = np.random.uniform(-1, 1, (2, 4))
    w2 = np.random.uniform(-1, 1, (4, 1))
    history = []
    
    for _ in range(epochs):
        loss, dw1, dw2 = get_grads(w1, w2)
        history.append(loss)
        
        g_flat = np.concatenate([dw1.flatten(), dw2.flatten()])
        gn2 = np.sum(g_flat**2)
        
        if gn2 < 1e-18: break 
        
        if mode == 'explicit':
            # Explicit is sensitive to lr/eps ratio
            mag = lr * (loss**alpha) / (gn2 + eps_val)
        else:
            # Implicit can handle much larger lr because of self-regulation
            # Term 'lr * loss^alpha' acts as a dynamic damping factor
            mag = lr * (loss**alpha) / (gn2 + lr * (loss**alpha))
            
        w1 -= mag * dw1
        w2 -= mag * dw2
        
        if np.isnan(loss) or loss > 1e2: return [np.nan] * epochs
            
    return history

# --- EXPERIMENT ---
alpha = 0.66 # Holder Regime
epochs = 1000

# Explicit needs a small LR to stay stable near the epsilon floor
loss_explicit = train(alpha, 'explicit', lr=0.02, eps_val=1e-5, epochs=epochs)

# Implicit uses an AGGRESSIVE LR to reach the target in fewer steps
loss_implicit = train(alpha, 'implicit', lr=0.5, epochs=epochs)

plt.figure(figsize=(10, 6))
plt.plot(loss_explicit, '--', color='red', label='Explicit (Standard η=0.02, ε=1e-5)')
plt.plot(loss_implicit, '-', color='blue', linewidth=2, label='Implicit (Aggressive η=0.5, ε=0)')

plt.yscale('log')
plt.title(f'XOR Performance: Explicit vs. Optimized Implicit (α={alpha})')
plt.xlabel('Iterations')
plt.ylabel('Loss (Log Scale)')
plt.legend()
#plt.grid(True, which="both", ls="-", alpha=0.2)
plt.show()
