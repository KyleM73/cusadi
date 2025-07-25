"""
From fork https://github.com/HariP19/cusadi
"""

import os, sys
EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(EXAMPLES_DIR)
sys.path.append(ROOT_DIR)

import torch
import numpy as np
import random
import time
import casadi as ca
import cusadi as cu
import cvxpy as cp

device = "cuda"
dtype = torch.double

# =================== Initialize the problem ===================
n = 3
NENV = 5
lin_problems = []
x_ground_truth = []
for i in range(NENV):
    A_i = np.random.randn(n, n)
    B_i = np.random.randn(n, 1)

    x_sol = np.linalg.solve(A_i, B_i)
    x_ground_truth.append(x_sol)

    lin_problems.append((A_i, B_i))

# solve using np.linalg.solve
lin_solve = ca.Function.load(os.path.join(cu.CUSADI_FUNCTION_DIR, "fn_lin_solve.casadi"))
fn_cusadi_lin_solve = cu.CusadiFunction(lin_solve, NENV)

# =================== Solve the problem using cusadi ===================
def parallel_lin_solve():
    # Flatten A to shape (NENV, 9) and B to shape (NENV, 3)
    A_flat = torch.stack([torch.tensor(p[0].T.reshape(-1), device=device, dtype=dtype) for p in lin_problems], dim=0)  # (NENV, 9)
    B_flat = torch.stack([torch.tensor(p[1].T.reshape(-1), device=device, dtype=dtype) for p in lin_problems], dim=0)  # (NENV, 3)

    # print("Shape of A_flat: ", A_flat.shape)
    # print("Shape of B_flat: ", B_flat.shape)
    fn_cusadi_lin_solve.evaluate([A_flat, B_flat])

    # The result is stored in outputs_sparse[0] as (NENV, 3)
    x_sol_cusadi = fn_cusadi_lin_solve.outputs_sparse[0]

    return x_sol_cusadi

# =================== Solve the problem using casadi ===================
def seq_lin_solve():
    x_sol_casadi = []
    for i in range(NENV):
        A = lin_problems[i][0]  # (3x3)
        B = lin_problems[i][1]  # (3x1)

        # Reshape A and B to match fn_lin_solve's inputs
        A_in = A.T.reshape(-1)  # (9x1)
        B_in = B.T.reshape(-1)    # (3x1)

        x_sol = lin_solve(A_in, B_in)
        x_sol_casadi.append(x_sol)
    return x_sol_casadi

# =================== Compare the results ===================
x_sol_cusadi = parallel_lin_solve().cpu().numpy()
x_sol_casadi = seq_lin_solve()

print("Cusadi solution:\n", x_sol_cusadi)
print("Casadi solution:\n", [x.full() for x in x_sol_casadi])
print("Ground truth:\n", x_ground_truth)
