import numpy as np
import pandas as pd
import requests
import polyline
import math
from scipy.optimize import minimize

# Python 3.14 compatibility issues
try:
    import srtm
    SRTM_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SRTM_AVAILABLE = False
    print("Warning: srtm module not available. Elevation data will be approximated as 0.")


MASS = 320.0             
BATTERY_CAP_WH = 3100.0  
MIN_SOC_LIMIT = 0.20     
PANEL_AREA = 6.0         
PANEL_EFF = 0.24         
CDA = 0.12              
CRR = 0.0025             
REGEN_EFF = 0.70  
MOTOR_EFF = 0.9     
RHO = 1.225             
G = 9.81                

# i have tried a global cache to prevent redundant simulations
sim_cache = {}


def get_route_data(start, end):
    print("Fetching route from OSRM...")
    url = f"http://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}?overview=full"
    res = requests.get(url).json()
    coords = polyline.decode(res['routes'][0]['geometry'])
    
    if SRTM_AVAILABLE:
        print("Querying SRTM elevation data...")
        elevation_data = srtm.get_data()
    else:
        print("Using mock elevation data (srtm unavailable)...")
        elevation_data = None
    
    data = []
    total_dist = 0
    for i in range(len(coords)):
        lat, lon = coords[i]
        if SRTM_AVAILABLE and elevation_data:
            alt = elevation_data.get_elevation(lat, lon)
        else:
            alt = 0  
        if i > 0:
            
            lat1, lon1 = coords[i-1]
            phi1, phi2 = math.radians(lat1), math.radians(lat)
            dphi, dlon = math.radians(lat-lat1), math.radians(lon-lon1)
            a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlon/2)**2
            step_dist = 2 * 6371000 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            total_dist += step_dist
            slope = math.asin((alt - data[i-1]['alt']) / step_dist) if step_dist > 0.1 else 0
        else:
            step_dist, slope = 0, 0
            
        data.append({'dist': step_dist, 'alt': alt, 'slope': slope})
    return pd.DataFrame(data)


def get_solar_power(t_seconds):
  
    irradiance = 1073 * np.exp(-((t_seconds - 43200)**2) / (2 * 11600**2))
    return irradiance * PANEL_AREA * PANEL_EFF

def run_full_day_sim(velocities, route_df, num_loops):
    current_time = 8 * 3600 
    energy_wh = BATTERY_CAP_WH * 1.0
    min_soc = 1.0
    
    
    chunks = np.array_split(route_df, len(velocities) - 1)
    
    
    for i, chunk in enumerate(chunks):
        v = velocities[i] / 3.6 ]
        for _, row in chunk.iterrows():
            dt = row['dist'] / v if v > 0 else 0
            p_req = ((0.5*RHO*CDA*v**3) + (CRR*MASS*G*np.cos(row['slope'])*v) + (MASS*G*np.sin(row['slope'])*v))/MOTOR_EFF
            p_in = get_solar_power(current_time)
            
            
            net_p = p_in + (abs(p_req) * REGEN_EFF) if p_req < 0 else p_in - p_req
            
            energy_wh = min(BATTERY_CAP_WH, energy_wh + (net_p * dt / 3600))
            min_soc = min(min_soc, energy_wh / BATTERY_CAP_WH)
            current_time += dt

    
    for _ in range(1800):
        energy_wh = min(BATTERY_CAP_WH, energy_wh + (get_solar_power(current_time) / 3600))
        current_time += 1

    
    v_loop = velocities[-1] / 3.6
    for _ in range(num_loops):
       
        loop_time = 35000 / v_loop
        for _ in range(int(loop_time)):
            p_loop = (0.5*RHO*CDA*v_loop**3) + (CRR*MASS*G*v_loop)
            energy_wh = min(BATTERY_CAP_WH, energy_wh + (get_solar_power(current_time) - p_loop)/3600)
            min_soc = min(min_soc, energy_wh / BATTERY_CAP_WH)
            current_time += 1
        
        for _ in range(300): 
            energy_wh = min(BATTERY_CAP_WH, energy_wh + (get_solar_power(current_time) / 3600))
            current_time += 1

    return current_time, min_soc, energy_wh / BATTERY_CAP_WH


def cached_sim(v, route_df, n):
    v_key = tuple(np.round(v, 3))
    if v_key not in sim_cache:
        sim_cache[v_key] = run_full_day_sim(v, route_df, n)
    return sim_cache[v_key]

def objective(v, route_df, n):
    
    finish_time, _, _ = cached_sim(v, route_df, n)
    return finish_time

def con_soc(v, route_df, n):
    _, min_soc, _ = cached_sim(v, route_df, n)
    return min_soc - MIN_SOC_LIMIT

def con_time(v, route_df, n):
    finish_time, _, _ = cached_sim(v, route_df, n)
    return (17 * 3600) - finish_time


if __name__ == "__main__":
  
    route = get_route_data((-26.816, 27.833), (-25.533, 26.083))
    
    best_overall_dist = 0
    final_plan = None

    
    for n_loops in range(0, 10):
        sim_cache.clear() 
        print(f"\nTesting feasibility for {n_loops} loops...")
        
       
        x0 = [65.0] * 11 
        res = minimize(
            objective, x0, args=(route, n_loops),
            method='SLSQP',
            bounds=[(35, 100)] * 11,
            constraints=[
                {'type': 'ineq', 'fun': con_soc, 'args': (route, n_loops)},
                {'type': 'ineq', 'fun': con_time, 'args': (route, n_loops)}
            ],
            options={'ftol': 1e-3}
        )
        
        if res.success:
            total_dist = (route['dist'].sum() / 1000) + (n_loops * 35)
            print(f"SUCCESS: {total_dist:.2f} km total distance.")
            if total_dist > best_overall_dist:
                best_overall_dist = total_dist
                final_plan = res.x
        else:
            print(f"FAILED: {n_loops} loops not feasible with current energy constraints.")
            break 

    print(f"\nOPTIMAL STRATEGY FOUND:")
    print(f"Total Distance: {best_overall_dist} km")
    print(f"Speed Profile (km/h): {np.round(final_plan, 2)}")