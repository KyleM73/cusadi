import os
import time
from typing import Callable
import torch
import casadi as ca
import cusadi as cu

class CusadiWrapper:
    def __init__(self, fn: ca.Function, num_envs: int) -> None:
        cu.generateCUDACodeDouble(fn)
        # cu.generatePytorchCode(fn)
        cu.generateCMakeLists(fn, dtype_str="double")
        t_compile = time.time()
        print("Compiling CUDA code...")
        status = os.system(f"cd {cu.CUSADI_ROOT_DIR} && rm -rf build && mkdir -p build && cd build && cmake .. && make -j")
        if status == 0:
            print("Compilation complete.")
        else:
            print("Compilation failed.")
            del self
            exit(1)
        t_compile = time.time() - t_compile
        print(f"Compilation time: {t_compile:.2f} seconds")
        self.fn = cu.CusadiFunction(fn, num_envs)
    
    def __call__(self, inputs: list) -> torch.Tensor:
        self.fn.evaluate(inputs)
        torch.cuda.synchronize() # necessary?
        return self.fn.getDenseOutput()

def cusadi(num_envs: int) -> Callable:
    def decorator(fn: ca.Function) -> Callable:
        cu.generateCUDACodeDouble(fn)
        # cu.generatePytorchCode(fn)
        cu.generateCMakeLists([fn], dtype_str="double")
        
        # Compile the CUDA code
        t_compile = time.time()
        print("Compiling CUDA code...")
        status = os.system(f"cd {cu.CUSADI_ROOT_DIR} && rm -rf build && mkdir -p build && cd build && cmake .. && make -j")
        if status != 0:
            raise RuntimeError("Compilation failed.")
        t_compile = time.time() - t_compile
        print(f"Compilation time: {t_compile:.2f} seconds")
        
        # Wrap the CasADi function with CusadiFunction
        cusadi_fn = cu.CusadiFunction(fn, num_envs)
        
        # Define the wrapped function
        def wrapped(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
            cusadi_fn.evaluate(inputs)
            torch.cuda.synchronize()  # Ensure GPU computations are complete
            return cusadi_fn.outputs_sparse
        
        return wrapped
    return decorator

# test
if __name__ == "__main__":

    def casadi_func() -> ca.Function:
        x = ca.SX.sym("x", 2) # type: ignore
        y = ca.SX.sym("y", 2) # type: ignore
        z = x + y
        return ca.Function("my_function", [x, y], [z])

    N = 1
    my_func = cusadi(N)(casadi_func())

    device = "cuda"
    inputs = [
        torch.rand(N, casadi_func().nnz_in(i), device=device, dtype=torch.double).contiguous()
            for i in range(casadi_func().n_in())
    ]
    outputs = my_func(inputs)
    print(inputs)
    print(outputs)