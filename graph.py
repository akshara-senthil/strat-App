import numpy as np
import matplotlib.pyplot as plt

# Constants (adjust these)
NLV = 1.0
eta = 0.8
k = 0.5
P_losses = 2.0
P_solar = 3.0

# Define function
def y(v):
    return (
            (NLV / eta) * (k * v**3 - (P_losses + eta * P_solar))
        )

# Generate v values
v = np.linspace(0.1, 10, 100)  # avoid zero for cube root issues

# Compute y
y_vals = y(v)

# Plot
plt.figure()
plt.plot(v, y_vals)
plt.xlabel("v")
plt.ylabel("y")
plt.title("Graph of the given function")
plt.grid()

plt.show()