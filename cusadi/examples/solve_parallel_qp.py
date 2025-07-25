"""
From fork https://github.com/HariP19/cusadi
"""

import os, sys
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
sys.path.append(ROOT_DIR)

import sys
import torch
from casadi import *
import numpy as np
import random
import time
from src import *
import cvxpy as cp

from src import CusadiFunction  # Make sure CusadiFunction is accessible
device = 'cuda'
dtype = torch.double

# Set paths and parameters
DATA_DIR = '/home/docker_dev/casadi_examples/optimization/test_data/mpc_qpbenchmark/data'
FILENAMES = [f"LIPMWALK{i}.npz" for i in range(3)]  # LIPMWALK0 to LIPMWALK29
N_PROBLEM = 100  # number of QP problems to solve in parallel
MAX_ITER = 10
TOL = 1e-4
rho_init = 1.0

# Load the penalty QP CasADi function
penalty_qp_step = Function.load(os.path.join(CUSADI_FUNCTION_DIR, "fn_penalty_qp_step.casadi"))
fn_cusadi_penalty_qp_step = CusadiFunction(penalty_qp_step, N_PROBLEM)

# Pre-load all QP problems into a list
qp_problems = []
for fname in FILENAMES:
    full_path = os.path.join(DATA_DIR, fname)
    data = np.load(full_path)
    P_prob = data['P']
    q_prob = data['q']
    G_prob = data['G']
    h_prob = data['h']
    data.close()
    qp_problems.append((P_prob, q_prob, G_prob, h_prob))

print("Random indices: ", random.choices(range(len(qp_problems)), k=N_PROBLEM))

# Randomly select N_PROBLEM QPs (with replacement if needed)
selected_problems = [qp_problems[i] for i in random.choices(range(len(qp_problems)), k=N_PROBLEM)]
# selected_problems = [qp_problems[i] for i in range(3)] # TODO: TEMP

# Extract dimensions
n = selected_problems[0][0].shape[0]  # dimension of decision variables from P
m = selected_problems[0][2].shape[0]  # number of inequality constraints from G

def solve_parallel_pqp():
    # Initialize all QPs as torch Tensors on the GPU
    x0_all = torch.randn((N_PROBLEM, n), device=device, dtype=dtype)
    lambda0_all = torch.zeros((N_PROBLEM, 1), device=device, dtype=dtype)
    mu0_all = torch.ones((N_PROBLEM, 1), device=device, dtype=dtype)

    Q_all = torch.stack([torch.tensor(p[0], device=device, dtype=dtype) 
                        for p in selected_problems], dim=0)
    q_all = torch.stack([torch.tensor(p[1], device=device, dtype=dtype) 
                        for p in selected_problems], dim=0)
    G_all = torch.stack([torch.tensor(p[2], device=device, dtype=dtype) 
                        for p in selected_problems], dim=0)
    h_all = torch.stack([torch.tensor(p[3], device=device, dtype=dtype) 
                        for p in selected_problems], dim=0)

    A_all = torch.zeros(N_PROBLEM, 0, n, device=device, dtype=dtype)
    b_all = torch.zeros((N_PROBLEM, 1), device=device, dtype=dtype)

    Q_all_flat = Q_all.transpose(1,2).reshape(N_PROBLEM, -1)
    A_all_flat = A_all.transpose(1,2).reshape(N_PROBLEM, -1)
    G_all_flat = G_all.transpose(1,2).reshape(N_PROBLEM, -1)

    start_time = time.time()
    for iter in range(MAX_ITER):

        # Evaluate the penalty QP step function
        fn_cusadi_penalty_qp_step.evaluate([x0_all, lambda0_all, mu0_all, Q_all_flat, q_all, A_all_flat, b_all, G_all_flat, h_all])

        # Retrieve results as torch tensors directly
        x_next = fn_cusadi_penalty_qp_step.outputs_sparse[0]
        lambda_next = fn_cusadi_penalty_qp_step.outputs_sparse[1]

        # Check convergence directly in torch
        dx = torch.norm(x_next - x0_all, dim=1)
        converged = dx < TOL

        if torch.all(converged):
            print(f'All QP problems converged in {iter} iterations in {time.time()-start_time:.2e} seconds')
            break

        # Update parameters for the next iteration
        x0_all = x_next.clone()
        # lambda0_all = lambda_next.clone() # Commented because no equality constraints
        mu0_all = mu0_all * 10

    if not torch.all(converged):
        print(f'Not all QP problems converged within {MAX_ITER} iterations')


    # print("\n-------------------Parallel PQP -------------------")    
    optimal_costs = []
    for i in range(N_PROBLEM):
        x = x_next[i, :]
        q = q_all[i, :]
        P = Q_all[i, :]
        optimal_costs.append(0.5 * x.T @ P @ x + q @ x)
        # print(f'Optimal x for QP problem {i}: {x}')
        print(f'Optimal cost for QP problem {i}: {optimal_costs[i]}\n')

# ## =================== Compare with cvxpy ===================
def solve_cvxpy(P, q, G, h, solver=cp.OSQP, verbose=False):
    n = P.shape[1]
    m_ineq = G.shape[0]
    x = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(x, P) + q.T @ x)

    constraints = [G @ x <= h]

    # Formulate and solve the problem
    problem = cp.Problem(objective, constraints)
    start = time.time()
    problem.solve(solver=solver, verbose=False)
    end = time.time()
    solve_time = end - start

    return x.value, solve_time, problem.status

# Solve the selected problems using parallel QP
solve_parallel_pqp()

# # Solve the selected problems using cvxpy
optimal_costs_cvxpy = []
solve_times = []
print("\n------------------- CVXPY -------------------")
for i in range(N_PROBLEM):
    P = selected_problems[i][0]
    q = selected_problems[i][1]
    G = selected_problems[i][2]
    h = selected_problems[i][3]
    x_cvxpy, solve_time, status = solve_cvxpy(P, q, G, h, verbose=False)
    solve_times.append(solve_time)
    optimal_costs_cvxpy.append(0.5 * x_cvxpy.T @ P @ x_cvxpy + q @ x_cvxpy)
    # print(f'Optimal x for QP problem {i} using cvxpy: {x_cvxpy}')
    print(f'Optimal cost for QP problem {i} using cvxpy: {optimal_costs_cvxpy[i]}\n')