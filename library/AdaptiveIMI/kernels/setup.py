import os
from pathlib import Path


os.environ.setdefault("CC", "gcc-11")
os.environ.setdefault("CXX", "g++-11")
os.environ.setdefault("CUDAHOSTCXX", "g++-11")

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SRC_DIR = HERE / "src"
CUTLASS_DIR = REPO_ROOT / "library" / "cutlass"

CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
CUDA_INCLUDE = os.path.join(CUDA_HOME, "include")

ext_modules = [
    CppExtension(
        "library.AdaptiveIMI.kernels.WaveBuffer",
        sources=[str(SRC_DIR / "wave_buffer_cpu.cpp")],
        include_dirs=[str(SRC_DIR), CUDA_INCLUDE],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
    CppExtension(
        "library.AdaptiveIMI.kernels.WaveBufferLRU",
        sources=[str(SRC_DIR / "wave_buffer_cpu_lru.cpp")],
        include_dirs=[str(SRC_DIR), CUDA_INCLUDE],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
    CppExtension(
        "library.AdaptiveIMI.kernels.WaveBufferLRUPlus",
        sources=[str(SRC_DIR / "wave_buffer_cpu_lru+.cpp")],
        include_dirs=[str(SRC_DIR), CUDA_INCLUDE],
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
    CUDAExtension(
        "library.AdaptiveIMI.kernels.Copy",
        sources=[str(SRC_DIR / "gather_copy.cu")],
        include_dirs=[str(SRC_DIR), CUDA_INCLUDE],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        },
        extra_link_args=["-lcuda", "-lcudart"],
    ),
    CUDAExtension(
        "library.AdaptiveIMI.kernels.gemm_softmax",
        sources=[str(SRC_DIR / "batch_gemm_softmax.cu")],
        include_dirs=[
            str(SRC_DIR),
            str(CUTLASS_DIR / "include"),
            str(CUTLASS_DIR / "examples" / "common"),
            str(CUTLASS_DIR / "tools" / "util" / "include"),
            CUDA_INCLUDE,
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": ["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        },
        extra_link_args=["-lcuda", "-lcudart"],
    ),
]


setup(
    name="adaptiveimi_kernels",
    version="0.1",
    packages=["library", "library.AdaptiveIMI", "library.AdaptiveIMI.kernels"],
    package_dir={"": str(REPO_ROOT)},
    description="AdaptiveIMI kernels and modules",
    long_description="A collection of CUDA and C++ extensions for AdaptiveIMI.",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    install_requires=["pybind11", "torch"],
    python_requires=">=3.10",
)
