import os
import time
from typing import Callable
import torch
import casadi as ca
import cusadi as cu

class CusadiWrapper:
    def __init__(self, fn_handle: Callable | ca.Function, num_envs: int, gen_pytorch: bool = False) -> None:
        if not isinstance(fn_handle, ca.Function):
            fn: ca.Function = fn_handle()
        else:
            fn: ca.Function = fn_handle
        cu.generateCUDACodeDouble(fn)
        if gen_pytorch:
            cu.generatePytorchCode(fn)
        cu.generateCMakeLists([fn], dtype_str="double")
        
        t_compile = time.time()
        print("Compiling CUDA code...")
        status = os.system(f"cd {cu.CUSADI_ROOT_DIR} && rm -rf build && mkdir -p build && cd build && cmake .. && make -j")
        if status != 0:
            raise RuntimeError(f"Compilation for function {fn.name()} failed.")
        t_compile = time.time() - t_compile
        print(f"Compilation time: {t_compile:.2f} seconds")
        self.fn = cu.CusadiFunction(fn, num_envs)
    
    def __call__(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        self.fn.evaluate(inputs)
        torch.cuda.synchronize()
        return self.fn.outputs_sparse

def cusadi_fn(num_envs: int) -> Callable:
    def decorator(fn_handle: Callable | ca.Function) -> Callable:
        if not isinstance(fn_handle, ca.Function):
            fn: ca.Function = fn_handle()
        else:
            fn: ca.Function = fn_handle
        cu.generateCUDACodeDouble(fn)
        # cu.generatePytorchCode(fn) # for easier debugging
        cu.generateCMakeLists([fn], dtype_str="double")
        
        t_compile = time.time()
        print("Compiling CUDA code...")
        status = os.system(f"cd {cu.CUSADI_ROOT_DIR} && rm -rf build && mkdir -p build && cd build && cmake .. && make -j")
        if status != 0:
            raise RuntimeError(f"Compilation for function {fn.name()} failed.")
        t_compile = time.time() - t_compile
        print(f"Compilation time: {t_compile:.2f} seconds")
        
        cusadi_fn = cu.CusadiFunction(fn, num_envs)
        
        def wrapped(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
            cusadi_fn.evaluate(inputs)
            torch.cuda.synchronize()
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

    mode = "decorator" # Options["decorator", "wrapper"]
    N = 5
    # decorator
    if mode == "decorator":
        my_func = cusadi_fn(num_envs=N)(casadi_func)
    # wrapper
    else:
        my_casadi_func = casadi_func()
        my_func = CusadiWrapper(my_casadi_func, num_envs=N)

    device = "cuda"
    inputs = [
        torch.rand(N, casadi_func().nnz_in(i), device=device, dtype=torch.double).contiguous()
            for i in range(casadi_func().n_in())
    ]
    outputs = my_func(inputs)
    print(inputs)
    print(outputs)
