"""
From fork https://github.com/HariP19/cusadi
"""

import os, sys
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
sys.path.append(ROOT_DIR)

import torch
import casadi as ca
import cusadi as cu
import numpy as np
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import time

# Simulation Parameters
N_ENV = 20 # number of environments
T = 50.0 # seconds
device = "cuda"
dtype = torch.double

# Pendulum Parameters
x0 = torch.rand((N_ENV, 2), device=device, dtype=dtype)
gravity = 9.81*torch.ones((N_ENV, 1), device=device, dtype=dtype)
l = torch.rand((N_ENV, 1), device=device, dtype=dtype)+0.5
dt = 0.01*torch.ones((N_ENV, 1), device=device, dtype=dtype)

# Load the CasADi function & create a CusadiFunction
fn_casadi_sim_step = ca.Function.load(os.path.join(cu.CUSADI_FUNCTION_DIR, "fn_sim_step.casadi"))
fn_cusadi_sim_step = cu.CusadiFunction(fn_casadi_sim_step, N_ENV)

def create_pendulum_vis(vis, step_size=1.0):
    # create a pendulum xy grid
    s = int(np.ceil(np.sqrt(N_ENV)))
    square_size = (s - 1) * step_size
    x = np.linspace(-square_size/2, square_size/2, s)
    y = np.linspace(-square_size/2, square_size/2, s)
    xv, yv = np.meshgrid(x, y)
    xy_points = np.vstack([xv.ravel(), yv.ravel()]).T[:N_ENV]

    for id in range(N_ENV):
        pendulum_name = f"pendulum{id}"
        rod_length = l[id].item()
        rod_radius = 0.01
        bob_radius = 0.05

        # create a pivot fixed 
        vis[pendulum_name]["pivot_fixed"].set_object(
            g.Box([0.05, 0.1, 0.05]),
            g.MeshLambertMaterial(color=0x00ff00)
        )
        vis[pendulum_name]["pivot_fixed"].set_transform(tf.translation_matrix([xy_points[id][0], xy_points[id][1], 1.5]))

        # Create pivot point
        T_init_pivot = tf.translation_matrix([0, 0, 0])
        R_init_pivot = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
        vis[pendulum_name]["pivot_fixed"]["pivot"].set_transform(R_init_pivot)

        # Create rod
        vis[pendulum_name]["pivot_fixed"]["pivot"]["rod"].set_object(
            g.Cylinder(height=rod_length, radius=rod_radius),
            g.MeshLambertMaterial(color=0x0000ff)
        )
        # Initial transform: rotate rod to point downward
        # R_init = tf.rotation_matrix(np.pi / 2, [0, 0, 1])
        R_init = tf.rotation_matrix(0, [1, 0, 0])
        T_init = tf.translation_matrix([0, -rod_length / 2 , 0])
        vis[pendulum_name]["pivot_fixed"]["pivot"]["rod"].set_transform(R_init @ T_init)

        # Create bob
        vis[pendulum_name]["pivot_fixed"]["pivot"]["rod"]["bob"].set_object(
            g.Sphere(radius=bob_radius),
            g.MeshLambertMaterial(color=0xff0000)
        )
        # Set initial position of bob at the end of the rod
        bob_position = np.array([0, -rod_length/2, 0])
        vis[pendulum_name]["pivot_fixed"]["pivot"]["rod"]["bob"].set_transform(tf.translation_matrix(bob_position))

def update_pendulum(vis, id, theta):
    pendulum_name = f"pendulum{id}"
    rod_length = l[id].item()

    # Compute rotation matrix for the rod
    R = tf.rotation_matrix(theta, [1, 0, 0])  # Rotate around z-axis
    R_init = tf.rotation_matrix(np.pi / 2, [1, 0, 0])
    transform = R @ R_init
    vis[pendulum_name]["pivot_fixed"]["pivot"].set_transform(transform)

def main(vis):
    # Create pendulum visualizations for all environments
    create_pendulum_vis(vis, step_size=3.5)

    # Simulate the pendulum
    num_steps = int(T/dt[0].item())

    # Measure the time taken for simulation
    start_time = time.time()
    current_state = x0
    trajectories = []
    for i in range(num_steps):
        fn_cusadi_sim_step.evaluate([current_state, gravity, l, dt])
        x_next = fn_cusadi_sim_step.outputs_sparse[0]
        current_state = x_next.clone()
        trajectories.append(x_next.cpu().numpy())
    end_time = time.time()
    print(f"Time taken for simulating {N_ENV} environments for {T} seconds: {end_time - start_time:.2f} seconds")

    trajectories = np.array(trajectories)
    trajectories_reshaped = np.transpose(trajectories, (1, 0, 2))

    # Visualization loop
    for step in range(num_steps):
        for id in range(N_ENV):
            theta = trajectories_reshaped[id][step, 0]  # Get the angle for the current step
            update_pendulum(vis, id, theta)  # Update the pendulum visualization
        time.sleep(0.01)  # Pause for the time step

vis = meshcat.Visualizer()
vis.open() 
main(vis)
