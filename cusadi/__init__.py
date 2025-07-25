import os, sys

CUSADI_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
CUSADI_CLASS_DIR = os.path.join(CUSADI_ROOT_DIR, "source")
CUSADI_FUNCTION_DIR = os.path.join(CUSADI_ROOT_DIR, "source", "casadi_functions")
CUSADI_BENCHMARK_DIR = os.path.join(CUSADI_ROOT_DIR, "source", "benchmark_functions")
CUSADI_BUILD_DIR = os.path.join(CUSADI_ROOT_DIR, "build")
CUSADI_CODEGEN_DIR = os.path.join(CUSADI_ROOT_DIR, "codegen")
CUSADI_DATA_DIR = os.path.join(CUSADI_ROOT_DIR, "../scripts/examples/data")

sys.path.insert(0, CUSADI_BUILD_DIR)

from .source import *
