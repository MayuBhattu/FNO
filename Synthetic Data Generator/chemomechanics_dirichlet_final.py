import matplotlib.pyplot as plt
import numpy as np
from firedrake import *
from tqdm import tqdm
import pandas as pd
import os
import time
from firedrake.output import VTKFile
os.environ["OMP_NUM_THREADS"] = "1"

# -----------------------------
# (2) MESH AND FUNCTION SPACES
# -----------------------------
# Square domain of size L x L, with Nx x Ny elements
L_dim = 500e-9 # Length of the square domain [m]
Nx, Ny = 20, 20
mesh = firedrake.UnitSquareMesh(Nx, Ny) # RectangleMesh(Nx, Ny, L, L)

# Function space for concentration (scalar)
V_c = FunctionSpace(mesh, "CG", 1)
# Function space for displacement (vector)
V_u = VectorFunctionSpace(mesh, "CG", 2)

# Create solution Functions

# Test and trial functions
v_c = TestFunction(V_c)
v_u = TestFunction(V_u)

# ------------------------------------
# (3) MODEL AND MATERIAL PARAMETERS
# ------------------------------------


##========= RANGE OF VALUES FOR BOUNDARY CONDITIONS ==============##
# J_dim - vary from 1.0e-6 to 1.0e-4
# c_max - vary from 1e3 to 1e5
##========= END VALUES FOR BOUNDARY CONDITIONS ===================##

D = 7.08e-15  # Diffusion coefficient [m2.s-1]
E_dim = 10e9     # Young's modulus [Pa]
nu_dim = 0.3     # Poisson's ratio
G_dim = E_dim / (2*(1 + nu_dim))     # Shear modulus [Pa]
Omega_dim = 3.497e-6 # Chemical expansion coefficient [m3.mol-1]
c_max = 2.95e5 # Maximum concentration [mol.m-3]

## Non-dimensional values

L = 1 # Dimensionless length of the square domain
E = E_dim / E_dim # Dimensionless Young's modulus
nu = nu_dim # Dimensionless Poisson's ratio
G = E / (2*(1 + nu)) # Dimensionless Shear modulus
Omega = Omega_dim * c_max # Chemical expansion coefficient

# Derived Lame parameters for plane strain
# lam = (E*nu)/((1+nu)*(1-2nu))  and  mu = G
lam = (E*nu)/((1+nu)*(1-2*nu))

# Coupling factor for diffusion-induced stress (from your PDE)
# alpha = (E * Omega) / (1 - 2 * nu)
alpha = (E*Omega)/(1 - 2*nu)

c_initial_values = np.logspace(0, 1, 10)  # 1e3 to 1e5, 5 values
c_bc_values = np.logspace(3, 5, 50)
J_dim_values = np.logspace(-6, -4, 50)  # 1e-6 to 1e-4, 5 values

### ===== ADDED BY SHAUNAK ===== ###
coordinates = mesh.coordinates.dat.data_ro
x, y = coordinates[:, 0], coordinates[:, 1]

# Create a structured array to preserve ordering
structured_array = np.array(list(zip(x, y)), dtype=[('x', float), ('y', float)])

# Sort first by y, then by x (row-wise sorting)
sorted_indices = np.lexsort((structured_array['x'], structured_array['y']))
sorted_coords = structured_array[sorted_indices]

# Sorted x and y
x_sorted = sorted_coords['x']
y_sorted = sorted_coords['y']
### ===== END BY SHAUNAK ===== ###

sim_id = 500
for c_initial_value in c_initial_values:
    for c_bc_value in c_bc_values:
        sim_id = sim_id + 1
        dt_value = 10   # time step [s]
        T_dim = 400     # final time [s]
        t_dim = 0.0    # initial time [s]

        dt_dim = dt_value

        dt = dt_dim * D / L_dim**2
        T = T_dim * D / L_dim**2
        t = t_dim * D / L_dim**2

        c = Function(V_c, name="Concentration")        # c^{n+1}
        c_old = Function(V_c, name="Concentration_old") # c^n
        u = Function(V_u, name="Displacement")

        # Set initial condition
        c_old.interpolate(c_initial_value/c_max)
        c.assign(c_old)
        
        c_bc = Constant(c_bc_value / c_max) # concentration boundary condition at boundary 4

        # ------------------------------------
        # (4) BOUNDARY CONDITIONS
        # ------------------------------------
        # Let’s assume edges are numbered 1,2,3,4 in Firedrake (depends on mesh ordering).
        # Zero displacement on edges 1,2,3 (Dirichlet).
        # Zero stress (traction-free) on edge 4 (no BC needed).

        # Zero displacement on edges 1,2,3
        zero_vec = Constant((0.0, 0.0))
        bcs_u = [
            DirichletBC(V_u, zero_vec, 1),
            DirichletBC(V_u, zero_vec, 2),
            DirichletBC(V_u, zero_vec, 3)
        ]
        # No displacement BC on edge 4 => traction-free by default

        # For concentration, you might set e.g. c=0 on all edges or partial edges if needed.
        # (Modify as appropriate for your problem.)
        bcs_c = [
           DirichletBC(V_c, c_bc, 4)
        ]

        # --------------------------------------
        # (5) DEFINE VARIATIONAL FORMS
        # --------------------------------------
        # 5.1) Diffusion:  c^{n+1} - c^n / dt + div(-D grad(c^{n+1})) = 0
        #     Backward-Euler: (c - c_old)/dt + div(J) = 0, J = -D grad(c)
        F_diff = ((c - c_old)/dt)*v_c*dx + dot(grad(c), grad(v_c))*dx

        # # Define the outward unit normal vector on the boundary
        # n = FacetNormal(mesh)
        # # Modify weak form to include Neumann BC on boundary 4
        # F_diff = ((c - c_old)/dt)*v_c*dx + dot(grad(c), grad(v_c))*dx - J_bc*v_c*ds(4)

        # 5.2) Mechanical Equilibrium:  div(sigma(u)) = alpha grad(c)
        #     Plane strain linear elasticity with chemical expansion.
        #     Stress: sigma(u) = 2*G*sym(grad(u)) + lam*tr(grad(u))*I
        #     PDE:   int(sigma(u) : grad(v_u) dx) - int(alpha grad(c) . v_u dx) = 0

        def sigma(u_):
            return 2.0*G*sym(grad(u_)) + lam*tr(sym(grad(u_)))*Identity(2)

        F_mech = inner(sigma(u), sym(grad(v_u)))*dx - alpha*dot(grad(c), v_u)*dx

        # ---------------------------
        # (6) TIME-STEPPING LOOP
        # ---------------------------
        # For each time step:
        #  - Solve diffusion for c^{n+1}
        #  - Solve mechanical equilibrium for u
        #  - Advance c_old = c
        #  - Increase time

        # Choose solvers (simple settings)
        diffusion_solver_params = {
            "ksp_type": "cg",
            "pc_type": "ilu"
        }
        mech_solver_params = {
            "ksp_type": "cg",
            "pc_type": "ilu"
        }
        
        df_list = []

        while t < T + 1e-9:
            # Solve diffusion
            solve(F_diff == 0, c, bcs = bcs_c,  solver_parameters=diffusion_solver_params)

            # Solve mechanics
            solve(F_mech == 0, u, bcs=bcs_u, solver_parameters=mech_solver_params)

            # Advance in time
            c_old.assign(c)
            t += dt

            def compute_stress(u):
                """Compute stress tensor components from displacement field."""
                eps = sym(grad(u))  # Strain tensor
                sigma_xx = project(2*G*eps[0, 0] + lam*tr(eps), V_c)
                sigma_yy = project(2*G*eps[1, 1] + lam*tr(eps), V_c)
                sigma_xy = project(2*G*eps[0, 1], V_c)  # Shear stress
                return sigma_xx.dat.data_ro, sigma_yy.dat.data_ro, sigma_xy.dat.data_ro

            sigma_xx, sigma_yy, sigma_xy = compute_stress(u)
            von_mises_stress = np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx * sigma_yy + 3 * sigma_xy**2)

            ### ===== ADDED BY SHAUNAK ===== ###
            # If you have solution values associated with the mesh nodes (e.g., c_values)
            c_values = c.dat.data_ro[sorted_indices]
            u_values = u.dat.data_ro[sorted_indices, 0]  # Displacement in x
            v_values = u.dat.data_ro[sorted_indices, 1]  # Displacement in y

            # Sort von Mises stress values using the same sorting order
            von_mises_stress = von_mises_stress[sorted_indices]
            ### ===== END BY SHAUNAK ===== ###

            df_list.append(pd.DataFrame({
                "c(x,y,t)": c_values,
                "stresses(x,y,t)": von_mises_stress,
                "time_step": t,
                "sim_id": sim_id  # Unique simulation ID
            }))
            #print(f"Time = {t:.2f} done.")

        final_df = pd.concat(df_list, ignore_index=True)
        output_dir = "simulation_results"
        os.makedirs(output_dir, exist_ok=True)  # Create folder if it doesn't exist

        # Save the final DataFrame in the specified folder
        file_path = os.path.join(output_dir, f"simulation_{sim_id}.csv")
        final_df.to_csv(file_path, index=False)

        print(sim_id)

        # fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        # #(5) Plot concentration profile
        # c_plot = axs[0].tricontourf(x_sorted, y_sorted, c_values, levels=50, cmap="plasma")

        # fig.colorbar(c_plot, ax=axs[0])
        # axs[0].set_title("Concentration Profile")
        # axs[0].set_xlabel("x")
        # axs[0].set_ylabel("y")

        # #(6) Plot von Mises stress
        # #stress_plot = axs[1].tricontourf(x, y, von_mises_stress, levels=50, cmap="jet")

        # fig.colorbar(stress_plot, ax=axs[1])
        # axs[1].set_title("Von Mises Stress")
        # axs[1].set_xlabel("x")
        # axs[1].set_ylabel("y")

        # plt.tight_layout()
        # plt.show()

