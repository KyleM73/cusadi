import os
import sys
import torch
import ctypes
import casadi as ca
from cusadi import CUSADI_CODEGEN_DIR, CUSADI_BUILD_DIR


class CusadiFunction:
    fn_casadi = None
    fn_name = None
    num_instances = 0
    inputs_sparse = []
    outputs_sparse = []
    outputs_dense = []

    _device = "cuda"
    _fn_library = None
    _work_tensor = []
    _input_tensors = []
    _input_ptrs = []
    _output_tensors = []
    _output_tensors_dense = []
    _output_ptrs = []
    _fn_input = []
    _fn_work = []
    _fn_output = []
    _dtype = None
    _ctype = None
    _ctype_str = None

    def __init__(self, fn_casadi, num_instances):
        assert torch.cuda.is_available()

        self.fn_casadi: ca.Function = fn_casadi
        self.fn_name = fn_casadi.name()
        self.num_instances = num_instances

        lib_filepath = os.path.join(CUSADI_BUILD_DIR, f"lib{self.fn_casadi.name()}.so")
        self._fn_library: ctypes.CDLL = ctypes.CDLL(lib_filepath)

        # Auto-detect precision
        self._ctype_str = self._detect_dtype()
        self._dtype = torch.float32 if self._ctype_str == "float" else torch.float64
        self._ctype = ctypes.c_float if self._ctype_str == "float" else ctypes.c_double

        self._fn_library.evaluate.restype = ctypes.c_float if self._detect_res_dtype() == "float" else None
        print("eval type: "+self._detect_res_dtype())

        print("Loaded CasADi function: ", self.fn_casadi)
        print("Loaded library: ", self._fn_library)
        print(f"Detected dtype: {self._dtype} ({self._ctype_str})")

        self._setup()

    def _detect_res_dtype(self) -> str:
        # Check generated source for float vs None kernel signature
        src_path = os.path.join(CUSADI_CODEGEN_DIR, f"{self.fn_name}.cu")
        try:
            with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
            if "float evaluate(const" in src:
                return "float"
            else:
                return "None"
        except Exception:
            pass
        print("Warning: could not detect dtype from source, defaulting to None.")
        return "None"

    def _detect_dtype(self) -> str:
        # Check generated source for float vs double kernel signature
        src_path = os.path.join(CUSADI_CODEGEN_DIR, f"{self.fn_name}.cu")
        try:
            with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                src = f.read()
            if "const float *inputs[]" in src:
                return "float"
            if "const double *inputs[]" in src:
                return "double"
        except Exception:
            pass
        print("Warning: could not detect dtype from source, defaulting to double.")
        return "double"

    def evaluate(self, inputs):
        self._clearTensors()
        self._prepareInputTensor(inputs)
        self.eval_time = self._fn_library.evaluate(
            self._fn_input,
            self._fn_work,
            self._fn_output,
            self.num_instances
        )
        return self.eval_time

    def getDenseOutput(self, out_idx=None):
        env_idx = torch.tensor(range(self.num_instances), device=self._device).repeat_interleave(
            self.fn_casadi.nnz_out(out_idx)
        )
        row_idx = torch.tensor(self.fn_casadi.sparsity_out(out_idx).get_triplet()[0], device=self._device) \
            .repeat(self.num_instances)
        col_idx = torch.tensor(self.fn_casadi.sparsity_out(out_idx).get_triplet()[1], device=self._device) \
            .repeat(self.num_instances)
        dim_dense = (
            self.num_instances,
            self.fn_casadi.size1_out(out_idx),
            self.fn_casadi.size2_out(out_idx)
        )
        return torch.sparse_coo_tensor(
            torch.vstack((env_idx, row_idx, col_idx)),
            self.outputs_sparse[out_idx].reshape(-1),
            dim_dense
        ).to_dense()

    def checkInputDimensions(self, inputs):
        self.input_CPU = [tensor[0, :].cpu().numpy() for tensor in inputs]
        try:
            _ = (self.fn_casadi.call(self.input_CPU)[0]).full()
            print("CPU call successful. Tensor dimensions are correct for inputs.")
        except Exception:
            print("Error in Casadi function call. Exiting...")
            sys.exit(1)

    def _setup(self):
        self._input_tensors = [
            torch.zeros((self.num_instances, self.fn_casadi.nnz_in(i)),
                        device=self._device, dtype=self._dtype).contiguous()
            for i in range(self.fn_casadi.n_in())
        ]
        self._output_tensors = [
            torch.zeros((self.num_instances, self.fn_casadi.nnz_out(i)),
                        device=self._device, dtype=self._dtype).contiguous()
            for i in range(self.fn_casadi.n_out())
        ]
        self._output_tensors_dense = [
            torch.zeros((self.num_instances,
                         self.fn_casadi.size1_out(i),
                         self.fn_casadi.size2_out(i)),
                        device=self._device, dtype=self._dtype).contiguous()
            for i in range(self.fn_casadi.n_out())
        ]
        self._work_tensor = torch.zeros(
            (self.num_instances, self.fn_casadi.sz_w()),
            device=self._device, dtype=self._dtype
        ).contiguous()

        self._input_ptrs = torch.zeros(self.fn_casadi.n_in(), device=self._device, dtype=torch.int64).contiguous()
        self._output_ptrs = torch.zeros(self.fn_casadi.n_out(), device=self._device, dtype=torch.int64).contiguous()

        for i in range(self.fn_casadi.n_in()):
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        for i in range(self.fn_casadi.n_out()):
            self._output_ptrs[i] = self._output_tensors[i].data_ptr()

        self._fn_input = self._castAsCPointer(self._input_ptrs.data_ptr(), "int")
        self._fn_output = self._castAsCPointer(self._output_ptrs.data_ptr(), "int")
        self._fn_work = self._castAsCPointer(self._work_tensor.data_ptr(), self._ctype_str)

        self.inputs_sparse = self._input_tensors
        self.outputs_sparse = self._output_tensors
        self.outputs_dense = self._output_tensors_dense

    def _prepareInputTensor(self, inputs):
        for i in range(self.fn_casadi.n_in()):
            self._input_tensors[i] = inputs[i]
            self._input_ptrs[i] = self._input_tensors[i].data_ptr()
        self._fn_input = self._castAsCPointer(self._input_ptrs.data_ptr(), "int")
        self.inputs_sparse = self._input_tensors

    def _clearTensors(self):
        for i in range(self.fn_casadi.n_out()):
            self._output_tensors[i].zero_()
        self._work_tensor.zero_()

    def _castAsCPointer(self, ptr, type="float"):
        if type == "int":
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_int))
        elif type == "float":
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float))
        elif type == "double":
            return ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double))
