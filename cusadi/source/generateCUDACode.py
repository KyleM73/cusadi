import os
import textwrap
import casadi as ca
import cusadi as cu

def generateCMakeLists(casadi_fns, dtype_str="double"):
    cmake_filepath = os.path.join(cu.CUSADI_ROOT_DIR, "CMakeLists.txt")
    cmake_file = open(cmake_filepath, "w+")
    cmake_strings = {}

    cmake_strings['version'] = "cmake_minimum_required(VERSION 3.15)\n"
    cmake_strings['project'] = "project(CusADi CXX CUDA)\n"
    cmake_strings['packages'] = textwrap.dedent(
    """
    # Find CUDA package
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
    message("CUDA found")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 86)
    endif()
    message("CUDA Architectures: " ${CMAKE_CUDA_ARCHITECTURES})

    # Set C++ standard
    set(CMAKE_CXX_STANDARD 11)

    # Set CUDA flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -arch=native --use_fast_math)  # Adjust architecture as needed

    """)

    # Set sources for each CasADi function
    str_libraries = ""
    str_sources = ""
    for f in casadi_fns:
        print(f.name())
        fn_source_name = f.name().upper() + "_SOURCE"
        fn_filepath = f"codegen/{f.name()}.cu"
        fn_lib_name = f"{f.name()}" # _{dtype_str}
        str_sources += f"set({fn_source_name} {fn_filepath})\n"
        str_libraries += f"add_library({fn_lib_name} SHARED ${{{fn_source_name}}})\n"
        str_libraries += f"target_link_libraries({fn_lib_name})\n"

    cmake_strings['sources'] = str_sources
    cmake_strings['include'] = textwrap.dedent(
    """
    # Include directories for your header files
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    """)

    # Add and link libraries for each CasADi function
    cmake_strings['libraries'] = str_libraries

    # Uncomment this to debug CUDA memory errors
    # If you do that, also comment out the the "Set CUDA flags" line above
    # cmake_strings['flags'] = textwrap.dedent(
    # f'''
    # target_compile_options({fn_lib_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    # -Xcompiler -rdynamic -lineinfo -arch=sm_89 --use_fast_math
    # >)
    # ''')
    # From fork https://github.com/ARCaD-Lab-UM/cusadi

    # * Write codegen to file
    for cmake_str in cmake_strings.values():
        cmake_file.write(cmake_str)

def generateCUDACodeDouble(f, filepath=None, benchmarking=True, debug_mode=True):
    print("Generating CUDA code for CasADi function: ", f.name())
    if filepath is None:
        codegen_filepath = os.path.join(cu.CUSADI_ROOT_DIR, "codegen", f"{f.name()}.cu")
    else:
        codegen_filepath = filepath
    codegen_file = open(codegen_filepath, "w+")
    codegen_strings = {}

    # * Parse CasADi function
    n_instr = f.n_instructions()
    n_in = f.n_in()
    n_out = f.n_out()
    nnz_in = [f.nnz_in(i) for i in range(n_in)]
    nnz_out = [f.nnz_out(i) for i in range(n_out)]
    n_w = f.sz_w()

    INSTR_LIMIT = n_instr

    input_idx = []
    input_idx_lengths = [0]
    output_idx = []
    output_idx_lengths = [0]
    for i in range(INSTR_LIMIT):
        input_idx.extend(f.instruction_input(i))
        input_idx_lengths.append(len(f.instruction_input(i)))
        output_idx.extend(f.instruction_output(i))
        output_idx_lengths.append(len(f.instruction_output(i)))
    operations = [f.instruction_id(i) for i in range(INSTR_LIMIT)]
    const_instr = [f.instruction_constant(i) for i in range(INSTR_LIMIT)]

    # * Codegen for const declarations and indices
    codegen_strings['header'] = "// AUTOMATICALLY GENERATED CODE FOR CUSADI\n"
    codegen_strings['includes'] = textwrap.dedent(
    '''
    #include <cuda_runtime.h>
    #include <math.h>
    #include <iostream>
    ''')
    codegen_strings["nnz_in"] = f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
    codegen_strings["nnz_out"] = f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
    codegen_strings["n_w"] = f"\n__constant__ int n_w = {n_w};\n"
    codegen_strings["error_check"] = textwrap.dedent(
    r'''
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    }
    ''')


    # * Codegen for CUDA kernel
    str_kernel = textwrap.dedent(
    '''

    __global__ void evaluate_kernel (
            const double *inputs[],
            double *work,
            double *outputs[],
            const int batch_size) {
    ''')
    str_kernel +=  "\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    str_kernel +=  "\n    int env_idx = idx * n_w;"
    str_kernel +=  "\n    if (idx < batch_size) {"
    o_instr = 0
    i_instr = 0
    for k in range(INSTR_LIMIT):
        op = operations[k]
        o_idx = output_idx[o_instr]
        i_idx = input_idx[i_instr]
        if op == ca.OP_CONST:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, const_instr[k])
        elif op == ca.OP_INPUT:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, i_idx, i_idx, input_idx[i_instr + 1])
        elif op == ca.OP_OUTPUT:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, o_idx, output_idx[o_instr + 1], i_idx)
        elif op == ca.OP_SQ:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, i_idx, i_idx)
        elif cu.OP_CUDA_DICT[op].count("%d") == 3:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, i_idx, input_idx[i_instr + 1])
        elif cu.OP_CUDA_DICT[op].count("%d") == 2:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, i_idx)
        else:
            raise Exception('Unknown CasADi operation: ' + str(op))
        o_instr += output_idx_lengths[k + 1]
        i_instr += input_idx_lengths[k + 1]

    str_kernel += "\n    }"         # End of if statement
    str_kernel += "\n}\n"           # End of kernel
    codegen_strings['cuda_kernel'] = str_kernel

    # * Codegen for C interface
    codegen_strings['c_interface_header'] = """\n\nextern "C" {\n"""
    codegen_strings['c_evaluation'] = textwrap.dedent(
    '''
        float evaluate(const double *inputs[],
                    double *work,
                    double *outputs[],
                    const int batch_size) {
            int blockSize = 256;
            int gridSize = (batch_size + blockSize - 1) / blockSize;
            float milliseconds;
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start, 0);
            evaluate_kernel<<<gridSize, blockSize>>>(inputs,
                                                    work,
                                                    outputs,
                                                    batch_size);
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&milliseconds, start, stop);
    ''')
    if debug_mode:
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaPeekAtLastError());"
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaDeviceSynchronize());"
    codegen_strings['c_evaluation'] += "\n    return milliseconds/1000;"
    codegen_strings['c_evaluation'] += "\n}"
    codegen_strings['c_interface_closer'] = """\n\n}"""

    # * Write codegen to file
    for cg_str in codegen_strings.values():
        codegen_file.write(cg_str)
    codegen_file.close()
    print("CUDA codegen complete for CasADi function: ", f.name())

def generateCUDACodeFloat(f, filepath=None, benchmarking=True, debug_mode=True):
    print("Generating CUDA code for CasADi function: ", f.name())
    if filepath is None:
        codegen_filepath = os.path.join(cu.CUSADI_ROOT_DIR, "codegen", f"{f.name()}.cu")
    else:
        codegen_filepath = filepath
    codegen_file = open(codegen_filepath, "w+")
    codegen_strings = {}

    # * Parse CasADi function
    n_instr = f.n_instructions()
    n_in = f.n_in()
    n_out = f.n_out()
    nnz_in = [f.nnz_in(i) for i in range(n_in)]
    nnz_out = [f.nnz_out(i) for i in range(n_out)]
    n_w = f.sz_w()

    INSTR_LIMIT = n_instr

    input_idx = []
    input_idx_lengths = [0]
    output_idx = []
    output_idx_lengths = [0]
    for i in range(INSTR_LIMIT):
        input_idx.extend(f.instruction_input(i))
        input_idx_lengths.append(len(f.instruction_input(i)))
        output_idx.extend(f.instruction_output(i))
        output_idx_lengths.append(len(f.instruction_output(i)))
    operations = [f.instruction_id(i) for i in range(INSTR_LIMIT)]
    const_instr = [f.instruction_constant(i) for i in range(INSTR_LIMIT)]

    # * Codegen for const declarations and indices
    codegen_strings['header'] = "// AUTOMATICALLY GENERATED CODE FOR CUSADI\n"
    codegen_strings['includes'] = textwrap.dedent(
    '''
    #include <cuda_runtime.h>
    #include <math.h>
    #include <iostream>
    ''')
    codegen_strings["nnz_in"] = f"\n__constant__ int nnz_in[] = {{{','.join(map(str, nnz_in))}}};"
    codegen_strings["nnz_out"] = f"\n__constant__ int nnz_out[] = {{{','.join(map(str, nnz_out))}}};"
    codegen_strings["n_w"] = f"\n__constant__ int n_w = {n_w};\n"
    codegen_strings["error_check"] = textwrap.dedent(
    r'''
    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
    }
    ''')


    # * Codegen for CUDA kernel
    str_kernel = textwrap.dedent(
    '''

    __global__ void evaluate_kernel (
            const float *inputs[],
            float *work,
            float *outputs[],
            const int batch_size) {
    ''')
    # str_kernel += f"\n    float work_env[{n_w}];"
    str_kernel +=  "\n    int idx = blockIdx.x * blockDim.x + threadIdx.x;"
    str_kernel +=  "\n    int env_idx = idx * n_w;"
    str_kernel +=  "\n    if (idx < batch_size) {"
    o_instr = 0
    i_instr = 0
    for k in range(INSTR_LIMIT):
        op = operations[k]
        o_idx = output_idx[o_instr]
        i_idx = input_idx[i_instr]
        if op == ca.OP_CONST:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, const_instr[k])
        elif op == ca.OP_INPUT:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, i_idx, i_idx, input_idx[i_instr + 1])
        elif op == ca.OP_OUTPUT:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, o_idx, output_idx[o_instr + 1], i_idx)
        elif op == ca.OP_SQ:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, i_idx, i_idx)
        elif cu.OP_CUDA_DICT[op].count("%d") == 3:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, i_idx, input_idx[i_instr + 1])
        elif cu.OP_CUDA_DICT[op].count("%d") == 2:
            str_kernel += cu.OP_CUDA_DICT[op] % (o_idx, i_idx)
        else:
            raise Exception('Unknown CasADi operation: ' + str(op))
        o_instr += output_idx_lengths[k + 1]
        i_instr += input_idx_lengths[k + 1]

    str_kernel += "\n    }"         # End of if statement
    str_kernel += "\n}\n"           # End of kernel
    codegen_strings['cuda_kernel'] = str_kernel

    # * Codegen for C interface
    codegen_strings['c_interface_header'] = """\n\nextern "C" {\n"""
    codegen_strings['c_evaluation'] = textwrap.dedent(
    '''
        void evaluate(const float *inputs[],
                    float *work,
                    float *outputs[],
                    const int batch_size) {
            int blockSize = 256;
            int gridSize = (batch_size + blockSize - 1) / blockSize;
            evaluate_kernel<<<gridSize, blockSize>>>(inputs,
                                                    work,
                                                    outputs,
                                                    batch_size);
    ''')
    if debug_mode:
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaPeekAtLastError());"
        codegen_strings['c_evaluation'] += "\n    gpuErrchk(cudaDeviceSynchronize());\n}"
    else:
        codegen_strings['c_evaluation'] += "\n}"
    codegen_strings['c_interface_closer'] = """\n\n}"""

    # * Write codegen to file
    for cg_str in codegen_strings.values():
        codegen_file.write(cg_str)
    codegen_file.close()
    print("CUDA codegen complete for CasADi function: ", f.name())
