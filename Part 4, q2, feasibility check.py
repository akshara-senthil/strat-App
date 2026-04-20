import numpy as np

# Parameters
T_max = 1.5        
L = 30             
E_current = 1.5    
E_min = 0.2        
P_solar = 250      
k = 0.005          # k can also be taken from the CRR value calculated in a prior question
eta = 0.90
P_losses = 50      
E_avail = E_current - E_min  

def check_feasible(N, v):
    t = N * L / v  # hours
    if t > T_max:
        return False, None, None
    net_power = (k * v**3 + P_losses) / eta  - P_solar            
    E_net = net_power * t / 1000           
    if E_net > E_avail:
        return False, None, None
    return True, t, E_net

# Search over N and v
best_N = 0
best_v = 0
results = {}

for N in range(1, 5):
    v_min_time = N * L / T_max   # minimum speed from time constraint
    feasible_vs = []
    for v in np.linspace(v_min_time, 150, 50000):
        ok, t, e = check_feasible(N, v)
        if ok:
            feasible_vs.append(v)
    if feasible_vs:
        results[N] = (min(feasible_vs), max(feasible_vs))
        if N > best_N:
            best_N = N
            best_v = min(feasible_vs)    # minimum feasible speed for max N

for N, (vlo, vhi) in results.items():
    print(f"N={N}: v ∈ [{vlo:.2f}, {vhi:.2f}] km/h")

print(f"\nMax loops: N = {best_N}")
print(f"Feasible speed range: {results[best_N][0]:.2f} – {results[best_N][1]:.2f} km/h")