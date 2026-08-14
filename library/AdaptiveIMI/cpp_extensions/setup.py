"""
统一构建脚本 - 编译 C++/CUDA 扩展模块

模块:
1. library.AdaptiveIMI.cpp_extensions.gpu_cluster_manager_cpp - IMI decode kernels
2. library.AdaptiveIMI.cpp_extensions.ultra_layer_pipeline_cpp - 32层并行K-means管道
3. library.AdaptiveIMI.cpp_extensions.Copy - gather/scatter CUDA kernels
4. library.AdaptiveIMI.cpp_extensions.hybrid_8way_builder_cpp - Hybrid 8-way GPU index builder

使用方法:
    cd cpp_extensions
    python setup.py build_ext --inplace
    # 或者
    pip install -e .
"""

import os
import subprocess
import re
import shutil
import site
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# 获取当前目录
HERE = Path(__file__).parent.absolute()
REPO_ROOT = HERE.parents[2]

# 获取 PyTorch 和 pybind11 的 include 路径
import torch
TORCH_INCLUDE = torch.utils.cpp_extension.include_paths()

# 获取 CUDA 路径
CUDA_HOME = os.environ.get('CUDA_HOME', '/usr/local/cuda')
CUDA_INCLUDE = os.path.join(CUDA_HOME, 'include')


def _existing_unique(paths):
    seen = set()
    result = []
    for path in paths:
        if not path:
            continue
        path = os.path.abspath(os.fspath(path))
        if path in seen or not os.path.exists(path):
            continue
        result.append(path)
        seen.add(path)
    return result


def _nvidia_package_dirs(kind):
    """Find headers/libs shipped by pip nvidia-* CUDA packages."""
    roots = []
    try:
        roots.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(site.getusersitepackages())
    except Exception:
        pass

    candidates = []
    for root in roots:
        nvidia_root = Path(root) / 'nvidia'
        if not nvidia_root.exists():
            continue
        for package_dir in nvidia_root.iterdir():
            candidate = package_dir / kind
            if candidate.exists():
                candidates.append(candidate)
    return _existing_unique(candidates)


NVIDIA_INCLUDE_DIRS = _nvidia_package_dirs('include')
NVIDIA_LIB_DIRS = _nvidia_package_dirs('lib')
CUDA_INCLUDE_DIRS = _existing_unique([
    CUDA_INCLUDE,
    os.path.join(CUDA_HOME, 'targets', 'x86_64-linux', 'include'),
] + NVIDIA_INCLUDE_DIRS)
CUDA_LIB_CANDIDATES = [
    os.path.join(CUDA_HOME, 'lib64'),
    os.path.join(CUDA_HOME, 'lib'),
    os.path.join(CUDA_HOME, 'targets', 'x86_64-linux', 'lib'),
    os.path.join(CUDA_HOME, 'lib64', 'stubs'),
    os.path.join(CUDA_HOME, 'targets', 'x86_64-linux', 'lib', 'stubs'),
] + NVIDIA_LIB_DIRS
for cuda_root in sorted(Path('/usr/local').glob('cuda-*')):
    CUDA_LIB_CANDIDATES.extend([
        str(cuda_root / 'lib64'),
        str(cuda_root / 'lib64' / 'stubs'),
        str(cuda_root / 'targets' / 'x86_64-linux' / 'lib'),
        str(cuda_root / 'targets' / 'x86_64-linux' / 'lib' / 'stubs'),
    ])
CUDA_LIB_DIRS = _existing_unique(CUDA_LIB_CANDIDATES)
if not CUDA_LIB_DIRS:
    CUDA_LIB_DIRS = [os.path.join(CUDA_HOME, 'lib64'), os.path.join(CUDA_HOME, 'lib')]

# 通用编译参数
COMMON_CXX_FLAGS = [
    '-O3',
    '-std=c++17',
    '-fPIC',
    '-fopenmp',
    '-march=native',
]

COMMON_NVCC_FLAGS = [
    '-O3',
    '-std=c++17',
    '--expt-relaxed-constexpr',
    '-lineinfo',
    '-allow-unsupported-compiler',
]

_host_compiler = os.environ.get('IMI_CUDA_HOST_COMPILER') or os.environ.get('CUDAHOSTCXX')
if not _host_compiler:
    _host_compiler = shutil.which('g++-11')
if _host_compiler:
    COMMON_NVCC_FLAGS.append(f'-ccbin={_host_compiler}')


def _parse_arch_list(arch_list_str):
    """Parse arch list like '8.0;8.9;9.0+PTX' to nvcc -gencode flags."""
    if not arch_list_str:
        return []

    tokens = re.split(r"[;,\s]+", arch_list_str.strip())
    flags = []
    for token in tokens:
        if not token:
            continue

        token_upper = token.upper()
        wants_ptx = token_upper.endswith("+PTX")
        base = token_upper.replace("+PTX", "")

        # Accept forms: 8.9 / 89
        if "." in base:
            parts = base.split(".")
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                continue
            cap = f"{parts[0]}{parts[1]}"
        elif base.isdigit():
            cap = base
        else:
            continue

        flags.append(f"-gencode=arch=compute_{cap},code=sm_{cap}")
        if wants_ptx:
            flags.append(f"-gencode=arch=compute_{cap},code=compute_{cap}")

    # Keep order, remove duplicates
    seen = set()
    unique = []
    for item in flags:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


# 检测 CUDA 架构
def get_cuda_arch_flags():
    """自动检测 GPU 架构"""
    # 0) User override has highest priority
    env_arch = os.environ.get('IMI_CUDA_ARCH_LIST') or os.environ.get('TORCH_CUDA_ARCH_LIST')
    env_flags = _parse_arch_list(env_arch)
    if env_flags:
        print(f"[setup.py] Using CUDA arch list from env: {env_arch}")
        return env_flags

    # 1) Try nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            caps = []
            for line in result.stdout.strip().split('\n'):
                cap = line.strip().replace('.', '').strip()
                if cap:
                    caps.append(f'-gencode=arch=compute_{cap},code=sm_{cap}')
            if caps:
                # Keep order while unique
                seen = set()
                unique_caps = []
                for item in caps:
                    if item not in seen:
                        unique_caps.append(item)
                        seen.add(item)
                return unique_caps
    except Exception:
        pass

    # 2) Try torch runtime (works even when nvidia-smi is unavailable)
    try:
        if torch.cuda.is_available():
            caps = []
            for idx in range(torch.cuda.device_count()):
                major, minor = torch.cuda.get_device_capability(idx)
                cap = f"{major}{minor}"
                caps.append(f'-gencode=arch=compute_{cap},code=sm_{cap}')
            if caps:
                seen = set()
                unique_caps = []
                for item in caps:
                    if item not in seen:
                        unique_caps.append(item)
                        seen.add(item)
                return unique_caps
    except Exception:
        pass

    # 3) Fallback: default common architectures
    return [
        '-gencode=arch=compute_70,code=sm_70',  # V100
        '-gencode=arch=compute_75,code=sm_75',  # T4 / RTX 20xx
        '-gencode=arch=compute_80,code=sm_80',  # A100
        '-gencode=arch=compute_86,code=sm_86',  # RTX 3090
        '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
        '-gencode=arch=compute_90,code=sm_90',  # H100/H800
    ]


CUDA_ARCH_FLAGS = get_cuda_arch_flags()

# ============================================================================
# 模块 1: gpu_cluster_manager_cpp (C++ + CUDA)
# ============================================================================
gpu_cluster_manager_sources = [
    str(HERE / 'src' / 'gpu_bindings.cpp'),
    str(HERE / 'src' / 'compute_wrappers.cpp'),
    str(HERE / 'src' / 'generated_page_packer.cpp'),
    str(HERE / 'cuda' / 'compute_impl.cu'),
]

gpu_cluster_manager_ext = CUDAExtension(
    name='library.AdaptiveIMI.cpp_extensions.gpu_cluster_manager_cpp',
    sources=gpu_cluster_manager_sources,
    include_dirs=[str(HERE / 'include')] + CUDA_INCLUDE_DIRS + TORCH_INCLUDE,
    library_dirs=CUDA_LIB_DIRS,
    extra_compile_args={
        'cxx': COMMON_CXX_FLAGS,
        'nvcc': COMMON_NVCC_FLAGS + CUDA_ARCH_FLAGS,
    },
)

# ============================================================================
# 模块 2: ultra_layer_pipeline_cpp (纯 C++，依赖 CUDA Runtime)
# ============================================================================
ultra_layer_pipeline_sources = [
    str(HERE / 'src' / 'pipeline_bindings.cpp'),
    str(HERE / 'src' / 'layer_pipeline.cpp'),
    str(HERE / 'src' / 'kmeans_core.cpp'),
    str(HERE / 'src' / 'reorganize_core.cpp'),
    str(HERE / 'src' / 'reorder_core.cpp'),
]

ultra_layer_pipeline_ext = CppExtension(
    name='library.AdaptiveIMI.cpp_extensions.ultra_layer_pipeline_cpp',
    sources=ultra_layer_pipeline_sources,
    include_dirs=[str(HERE / 'include')] + CUDA_INCLUDE_DIRS + TORCH_INCLUDE,
    library_dirs=CUDA_LIB_DIRS,
    libraries=['cudart'],
    extra_compile_args={
        'cxx': COMMON_CXX_FLAGS,
    },
    extra_link_args=['-fopenmp'],
)

# 模块 3: Copy (CUDA)
# ============================================================================
copy_ext = CUDAExtension(
    name='library.AdaptiveIMI.cpp_extensions.Copy',
    sources=[str(HERE / 'cuda' / 'gather_copy.cu')],
    include_dirs=[str(HERE / 'include')] + CUDA_INCLUDE_DIRS + TORCH_INCLUDE,
    library_dirs=CUDA_LIB_DIRS,
    extra_compile_args={
        'cxx': COMMON_CXX_FLAGS,
        'nvcc': COMMON_NVCC_FLAGS + CUDA_ARCH_FLAGS,
    },
    extra_link_args=['-lcuda', '-lcudart'],
)

# ============================================================================
# 模块 4: hybrid_8way_builder_cpp (CUDA)
# ============================================================================
hybrid_8way_builder_sources = [
    str(HERE / 'src' / 'hybrid_8way_bindings.cpp'),
    str(HERE / 'src' / 'reorder_core.cpp'),
    str(HERE / 'cuda' / 'hybrid_8way_builder.cu'),
]

hybrid_8way_builder_ext = CUDAExtension(
    name='library.AdaptiveIMI.cpp_extensions.hybrid_8way_builder_cpp',
    sources=hybrid_8way_builder_sources,
    include_dirs=[str(HERE / 'include')] + CUDA_INCLUDE_DIRS + TORCH_INCLUDE,
    library_dirs=CUDA_LIB_DIRS,
    extra_compile_args={
        'cxx': COMMON_CXX_FLAGS,
        'nvcc': COMMON_NVCC_FLAGS + CUDA_ARCH_FLAGS,
    },
    extra_link_args=['-lcuda', '-lcudart'],
)

# ============================================================================
# Setup
# ============================================================================
_all_extensions = [
    gpu_cluster_manager_ext,
    ultra_layer_pipeline_ext,
    copy_ext,
    hybrid_8way_builder_ext,
]
_requested_modules = {
    item.strip() for item in os.environ.get("IMI_BUILD_MODULES", "").split(",") if item.strip()
}
if _requested_modules:
    _extension_aliases = {
        "gpu_cluster_manager_cpp": gpu_cluster_manager_ext,
        "ultra_layer_pipeline_cpp": ultra_layer_pipeline_ext,
        "Copy": copy_ext,
        "hybrid_8way_builder_cpp": hybrid_8way_builder_ext,
    }
    unknown_modules = _requested_modules.difference(_extension_aliases)
    if unknown_modules:
        raise RuntimeError(f"unknown IMI_BUILD_MODULES entries: {sorted(unknown_modules)}")
    _all_extensions = [
        extension for alias, extension in _extension_aliases.items() if alias in _requested_modules
    ]

setup(
    name='imi_cpp_extensions',
    version='1.0.0',
    packages=['library', 'library.AdaptiveIMI', 'library.AdaptiveIMI.cpp_extensions'],
    package_dir={'': str(REPO_ROOT)},
    description='IMI Cache C++/CUDA Extensions',
    ext_modules=_all_extensions,
    cmdclass={
        'build_ext': BuildExtension,
    },
    python_requires='>=3.8',
)
