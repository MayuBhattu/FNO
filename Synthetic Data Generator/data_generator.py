import matplotlib.pyplot as plt
import numpy as np
from firedrake import *
from tqdm import tqdm
import pandas as pd
import os
import time
from firedrake.output import VTKFile

os.environ["OMP_NUM_THREADS"] = "1"
# ============ PARAMETERS ============
T = 10             # Final time
num_steps = 200    # Number of time steps
dt = T / num_steps # Time step size
num_points = 20    # Grid resolution
num_simulations = 1000  # Number of different PDE simulations

# Define grid points for sampling
x_vals = np.linspace(0, 1, num_points+1)
y_vals = np.linspace(0, 1, num_points+1)
X, Y = np.meshgrid(x_vals, y_vals)
sample_points = np.vstack([X.ravel(), Y.ravel()]).T  # Shape: (num_points * num_points, 2)

# Create directory for storing data
output_dir = "./simulation_results_new/"
os.makedirs(output_dir, exist_ok=True)

# ============ CREATE MESH & FUNCTION SPACE ONCE ============
mesh = UnitSquareMesh(num_points, num_points)
V = FunctionSpace(mesh, 'CG', 2)
x, y = SpatialCoordinate(mesh)

sim_bar = tqdm(total=num_simulations, desc="Total Simulations", position=0, leave=True, dynamic_ncols=True)
# ============ SIMULATION LOOP ============
for sim in (range(num_simulations)):
    np.random.seed(sim) 
    if (sim+1)%100==0:  # Replace with your actual condition
        print("Pausing execution for 5 minutes...")
        time.sleep(5 * 60)  # 5 minutes = 300 seconds
        print("Resuming execution!")
    # ============ RANDOMLY ASSIGN BOUNDARY CONDITIONS ============
    boundary_types = {  # Each boundary gets a random BC type
        1: np.random.choice(["Dirichlet", "Neumann"]),  # Left (x = 0)
        2: np.random.choice(["Dirichlet", "Neumann"]),  # Right (x = 1)
        3: np.random.choice(["Dirichlet", "Neumann"]),  # Bottom (y = 0)
        4: np.random.choice(["Dirichlet", "Neumann"])   # Top (y = 1)
    }

    boundary_values = {  # Random values for BCs
        1: np.random.uniform(-0.5, 0.5),  # Left
        2: np.random.uniform(-0.5, 0.5),  # Right
        3: np.random.uniform(-0.5, 0.5),  # Bottom
        4: np.random.uniform(-0.5, 0.5)   # Top
    }

    # ============ DIRECTLY MODIFY NUMPY ARRAYS ============
    # Initialize NumPy arrays for storing boundary information
    dirichlet_mask_array = np.zeros(sample_points.shape[0])  # Default: no Dirichlet BC
    neumann_mask_array = np.zeros(sample_points.shape[0])    # Default: no Neumann BC
    bc_value_array = np.zeros(sample_points.shape[0])        # Default: zero BC values

    # Loop through the boundaries and apply BCs
    for boundary, bc_type in boundary_types.items():
        value = boundary_values[boundary]

        # Identify points on the boundary based on coordinates
        if boundary == 1:  # Left boundary (x = 0)
            mask = sample_points[:, 0] == 0
        elif boundary == 2:  # Right boundary (x = 1)
            mask = sample_points[:, 0] == 1
        elif boundary == 3:  # Bottom boundary (y = 0)
            mask = sample_points[:, 1] == 0
        elif boundary == 4:  # Top boundary (y = 1)
            mask = sample_points[:, 1] == 1

        # Assign values based on BC type
        if bc_type == "Dirichlet":
            dirichlet_mask_array[mask] = 1
            bc_value_array[mask] = value
        elif bc_type == "Neumann":
            neumann_mask_array[mask] = 1
            bc_value_array[mask] = value

    # ============ RANDOM INITIAL CONDITION ============
    u_n = Function(V).interpolate(Constant(np.random.uniform(0, 1)))

    # ============ STORE STATIC DATA ============
    sampled_initial_condition = np.array([u_n.at(p) for p in sample_points])

    static_data = pd.DataFrame({
        "x": sample_points[:, 0],
        "y": sample_points[:, 1],
        "Initial_Condition": sampled_initial_condition,  
        "Dirichlet_Mask": dirichlet_mask_array,
        "Neumann_Mask": neumann_mask_array,
        "BC_Value": bc_value_array,
        "sim_id": sim  # Unique simulation ID
    })

    # Save static data **immediately** so it's not lost
    static_data.to_csv(f"{output_dir}/static_data_sim{sim}.csv", index=False)

    # ============ PDE FORMULATION ============
    u = Function(V)
    v = TestFunction(V)

    # Weak form for diffusion + Neumann BCs
    F = (
        (u - u_n) * v * dx
        + dt * inner(grad(u), grad(v)) * dx
        - sum(dt * boundary_values[b] * v * ds(b) for b in boundary_types if boundary_types[b] == "Neumann")
    )

    # ============ TIME STEPPING ============
    t = 0

    # Create list to store time-dependent data
    time_series_data = []


    for n in (range(num_steps)):
        solve(F == 0, u, bcs=[DirichletBC(V, Constant(boundary_values[b]), b) for b in boundary_types if boundary_types[b] == "Dirichlet"])
        t += dt
        u_n.assign(u)

        #sampled_values = u_proj.dat.data
        sampled_values = np.array([u.at(np.array(p, dtype=np.float64, copy=True)) for p in sample_points])
        df = pd.DataFrame({
            "u(x,y,t)": sampled_values,
            "time_step": n,
            "sim_id": sim  # Unique simulation ID
        })
        time_series_data.append(df)

    # Save time-series data **immediately** after each simulation
    full_data = pd.concat(time_series_data, ignore_index=True)
    full_data.to_csv(f"{output_dir}/time_series_data_sim{sim}.csv", index=False)

    sim_bar.update(1)  # Update total simulation count

print("✅ All simulations saved successfully!")