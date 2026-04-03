#!/bin/bash
# ============================================================================
# 统一构建脚本 - 一次编译三个C++/CUDA扩展模块
# ============================================================================
#
# 模块:
#   1. cluster_manager_cpp - CPU端元数据和位置管理
#   2. gpu_cluster_manager_cpp - GPU端存储和计算
#   3. ultra_layer_pipeline_cpp - 32层并行K-means管道
#
# 使用方法:
#   cd cpp_extensions
#   bash build.sh
#
# ============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=============================================="
echo "  IMI C++/CUDA Extensions - Unified Build"
echo "=============================================="
echo ""

# 优先使用 conda 环境中的 CUDA 库，避免与系统 CUDA 冲突
if [ -n "$CONDA_PREFIX" ]; then
    if [ -d "$CONDA_PREFIX/lib" ]; then
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
    fi
    if [ -d "$CONDA_PREFIX/lib64" ]; then
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib64:${LD_LIBRARY_PATH}"
    fi
fi

# 检查 CUDA - 优先使用 conda 中的 nvcc（若存在）
if [ -z "$CUDA_HOME" ]; then
    if [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/bin/nvcc" ]; then
        export CUDA_HOME="$CONDA_PREFIX"
    elif [ -d "/usr/local/cuda-12.4" ]; then
        export CUDA_HOME=/usr/local/cuda-12.4
    elif [ -d "/usr/local/cuda-12.2" ]; then
        export CUDA_HOME=/usr/local/cuda-12.2
    elif [ -d "/usr/local/cuda-12" ]; then
        export CUDA_HOME=/usr/local/cuda-12
    elif [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    else
        echo "❌ CUDA_HOME not set and CUDA not found"
        exit 1
    fi
fi
echo "✅ CUDA_HOME: $CUDA_HOME"

# 设置 LD_LIBRARY_PATH
if [ -d "$CUDA_HOME/lib64" ]; then
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CUDA_HOME/lib64"
fi

# 清理旧的构建文件
echo ""
echo "🧹 Cleaning old build files..."
rm -rf build/ dist/ *.egg-info/
rm -f *.so

# 构建
echo ""
echo "🔨 Building extensions..."
python setup.py build_ext --inplace 2>&1 | tee build.log

# 检查构建结果
echo ""
echo "=============================================="
echo "  Build Results"
echo "=============================================="

BUILD_SUCCESS=true

if [ -f "cluster_manager_cpp.cpython-"*".so" ]; then
    echo "✅ cluster_manager_cpp: $(ls cluster_manager_cpp.cpython-*.so)"
else
    echo "❌ cluster_manager_cpp: FAILED"
    BUILD_SUCCESS=false
fi

if [ -f "gpu_cluster_manager_cpp.cpython-"*".so" ]; then
    echo "✅ gpu_cluster_manager_cpp: $(ls gpu_cluster_manager_cpp.cpython-*.so)"
else
    echo "❌ gpu_cluster_manager_cpp: FAILED"
    BUILD_SUCCESS=false
fi

if [ -f "ultra_layer_pipeline_cpp.cpython-"*".so" ]; then
    echo "✅ ultra_layer_pipeline_cpp: $(ls ultra_layer_pipeline_cpp.cpython-*.so)"
else
    echo "❌ ultra_layer_pipeline_cpp: FAILED"
    BUILD_SUCCESS=false
fi

echo ""

if [ "$BUILD_SUCCESS" = true ]; then
    echo "=============================================="
    echo "  ✅ All modules built successfully!"
    echo "=============================================="
    echo ""
    echo "测试导入:"

    # 添加 PyTorch 库路径到 LD_LIBRARY_PATH
    TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
    export LD_LIBRARY_PATH="${TORCH_LIB_PATH}:${LD_LIBRARY_PATH}"

    python -c "
import cluster_manager_cpp as cm
import gpu_cluster_manager_cpp as gmc
import ultra_layer_pipeline_cpp as ulp
print('  cluster_manager_cpp:', dir(cm)[:3], '...')
print('  gpu_cluster_manager_cpp:', dir(gmc)[:3], '...')
print('  ultra_layer_pipeline_cpp:', dir(ulp)[:3], '...')
print('✅ All imports successful!')
"
else
    echo "=============================================="
    echo "  ❌ Build failed! Check build.log for details"
    echo "=============================================="
    exit 1
fi
