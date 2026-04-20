import numpy as np
import matplotlib.pyplot as plt

def f(v, N):
    return (900*v**3) - 125 - 182*v/N

v = np.linspace(-10, 10, 1000)

N_values = [1, 2, 3, 5, 10, 20]
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

fig, ax = plt.subplots(figsize=(11, 7))

for N, color in zip(N_values, colors):
    y = f(v, N)
    ax.plot(v, y, label=f'N = {N}', color=color, linewidth=2)
    # Shade region where f(v) <= 0
    ax.fill_between(v, y, 0, where=(y <= 0), alpha=0.08, color=color)

ax.axhline(y=0, color='black', linewidth=1.5, linestyle='--', label='y = 0')
ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')

ax.set_xlabel('v', fontsize=13)
ax.set_ylabel('f(v)', fontsize=13)
ax.set_title(r'$\frac{15v^3}{70} - \frac{750}{7} - \frac{1.3v}{30N} \leq 0$ for different N',
             fontsize=14)
ax.legend(fontsize=11)
ax.set_ylim(-300, 300)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cubic_inequality_v2.png', dpi=150)
plt.show()

# Print roots for each N
print("Roots (where f(v) = 0) for each N:")
for N in N_values:
    y = f(v, N)
    sign_changes = np.where(np.diff(np.sign(y)))[0]
    roots = [round(float(v[i]), 3) for i in sign_changes]
    print(f"  N = {N:>2}: v ≈ {roots}")
    
    # Print solution to inequality f(v) <= 0
    sol = v[y <= 0]
    if len(sol) > 0:
        print(f"         f(v)≤0 for v ≤ {round(float(sol[-1]), 3)}")