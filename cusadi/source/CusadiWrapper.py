import os
import time
from typing import Callable, Literal
import torch
import casadi as ca
import cusadi as cu


class CusadiWrapper:
    def __init__(
            self,
            fn_handle: Callable | ca.Function | str,
            num_envs: int,
            dtype_str: Literal["double", "float"] = "float",
            gen_pytorch: bool = False,
            lib_exists: bool = False,
            benchmark_mode: bool = False,
            **kwargs,
    ) -> None:
        if isinstance(fn_handle, str):
            loaded_fn: ca.Function = ca.Function.load(fn_handle)  # type: ignore
        elif not isinstance(fn_handle, ca.Function):
            loaded_fn: ca.Function = fn_handle()  # type: ignore
        else:
            loaded_fn: ca.Function = fn_handle
        self.casadi_fn = loaded_fn
        self.n = num_envs
        self.dtype_str: Literal["float", "double"] = dtype_str
        self.dtype = torch.float if self.dtype_str == "float" else torch.double
        self.gen_pytorch = gen_pytorch
        if not lib_exists:
            self._codegen(benchmark_mode)
        self.fn = cu.CusadiFunction(self.casadi_fn, self.n)
        self.eval_time = None

    def _codegen(self, benchmark_mode: bool) -> None:
        """Generate CUDA code for the CasADi function."""
        cu.generateCUDACode(
            self.casadi_fn,
            dtype_str=self.dtype_str,
            benchmark_mode=benchmark_mode,
            debug_mode=False
        )
        if self.gen_pytorch:
            cu.generatePytorchCode(self.casadi_fn)
        cu.generateCMakeLists([self.casadi_fn])

        t_compile = time.time()
        print("Compiling CUDA code...")
        status = os.system(
            f"cd {cu.CUSADI_ROOT_DIR} && rm -rf build && mkdir -p build "
            f"&& cd build && cmake .. && make -j"
        )
        if status != 0:
            raise RuntimeError(f"Compilation for function {self.casadi_fn.name()} failed.")
        t_compile = time.time() - t_compile
        print(f"Compilation time: {t_compile:.2f} seconds")

    def __call__(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        self.eval_time = self.fn.evaluate(inputs)  # returns eval_time if benchmark_mode == True
        torch.cuda.synchronize()
        return self.fn.outputs_sparse

    def get_dummy_inputs(self) -> list[torch.Tensor]:
        return [
            torch.rand(
                self.n, self.casadi_fn.nnz_in(i), device="cuda", dtype=self.dtype
            ).contiguous()
            for i in range(self.casadi_fn.n_in())
        ]

    def input_size(self, i: int) -> int:
        return self.casadi_fn.nnz_in(i)

    def output_size(self, i: int) -> int:
        return self.casadi_fn.nnz_out(i)

    @property
    def n_inputs(self) -> int:
        return self.casadi_fn.n_in()

    @property
    def n_outputs(self) -> int:
        return self.casadi_fn.n_out()

    @property
    def num_envs(self) -> int:
        return self.n

    def get_eval_time(self) -> float | None:
        return self.eval_time

    def __repr__(self) -> str:
        return f"{self.casadi_fn.name()}_cusadi"


def cusadi_fn(
        num_envs: int,
        dtype_str: Literal["double", "float"] = "float",
        gen_pytorch: bool = False,
) -> Callable:
    def decorator(fn_handle: Callable | ca.Function) -> Callable:
        if not isinstance(fn_handle, ca.Function):
            casadi_fn: ca.Function = fn_handle()  # type: ignore
        else:
            casadi_fn: ca.Function = fn_handle
        cu.generateCUDACode(
            casadi_fn,
            dtype_str=dtype_str,
            benchmark_mode=False,
            debug_mode=False
        )
        if gen_pytorch:
            cu.generatePytorchCode(casadi_fn)
        cu.generateCMakeLists([casadi_fn])

        t_compile = time.time()
        print("Compiling CUDA code...")
        status = os.system(
            f"cd {cu.CUSADI_ROOT_DIR} && rm -rf build && mkdir -p build "
            f"&& cd build && cmake .. && make -j"
        )
        if status != 0:
            raise RuntimeError(f"Compilation for function {casadi_fn.name()} failed.")
        t_compile = time.time() - t_compile
        print(f"Compilation time: {t_compile:.2f} seconds")

        cusadi_function = cu.CusadiFunction(casadi_fn, num_envs)

        def wrapped(inputs: list[torch.Tensor]) -> list[torch.Tensor]:
            cusadi_function.evaluate(inputs)
            torch.cuda.synchronize()
            return cusadi_function.outputs_sparse

        return wrapped
    return decorator


# test
if __name__ == "__main__":
    def casadi_func() -> ca.Function:
        x = ca.SX.sym("x", 2)  # type: ignore
        y = ca.SX.sym("y", 2)  # type: ignore
        z = x + y
        return ca.Function("my_function", [x, y], [z])

    mode = "wrapper"  # Options["decorator", "wrapper"]
    N = 1
    dtype_str = "float"  # Options["double", "float"]
    device = "cuda"
    benchmark_mode = True
    # decorator
    if mode == "decorator":
        my_func = cusadi_fn(num_envs=N, dtype_str=dtype_str)(casadi_func)
        inputs = [
            torch.rand(
                N, casadi_func().nnz_in(i), device=device,
                dtype=torch.float if dtype_str == "float" else torch.double
            ).contiguous()
            for i in range(casadi_func().n_in())
        ]
    # wrapper
    else:
        my_func = CusadiWrapper(casadi_func, num_envs=N, dtype_str=dtype_str, benchmark_mode=benchmark_mode)
        inputs = my_func.get_dummy_inputs()

    outputs = my_func(inputs)
    print(inputs)
    print(outputs)
    if mode == "wrapper" and benchmark_mode:
        print(f"Evaluation: {my_func.get_eval_time():4f}s")
