import os
import time
from typing import Callable
import torch
import casadi as ca
import cusadi as cu

class CusadiWrapper:
    def __init__(self, fn: ca.Function, num_envs: int) -> None:
        cu.generateCUDACodeDouble(fn)
        cu.generatePytorchCode(fn)
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
        cu.generatePytorchCode(fn)
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
        def wrapped(inputs: list) -> torch.Tensor:
            cusadi_fn.evaluate(inputs)
            torch.cuda.synchronize()  # Ensure GPU computations are complete
            return cusadi_fn.getDenseOutput(out_idx=0)
        
        return wrapped
    return decorator

# test
if __name__ == "__main__":

    def casadi_func() -> ca.Function:
        x = ca.SX.sym("x", 2) # type: ignore
        # y = ca.SX.sym("y", 2) # type: ignore
        z = x + 1 # + y
        # return ca.Function("my_function", [x, y], [z])
        return ca.Function("my_function", [x], [z])

    # casadi_fn = casadi_func()
    # casadi_inputs = [ca.DM([1.0, 2.0]), ca.DM([5.0, 6.0])]
    # casadi_output = casadi_fn(*casadi_inputs)
    # print("CasADi output:", casadi_output)

    my_func = cusadi(1)(casadi_func())

    device = "cuda"
    # inputs = [
    #     torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device),
    #     torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device),
    # ]
    inputs = [
        torch.tensor([[1.0, 2.0]], device=device).view(2, 1),
        torch.tensor([[5.0, 6.0]], device=device).view(2, 1),
    ]
    output = my_func([inputs[0]])
    print(output)