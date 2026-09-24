"""Build the cckernel CUDA extension (Ada / sm_89 only).

    pip install -e .            # or: python setup.py build_ext --inplace
Set CCK_ARCH to override the target (default "89").
"""

import glob
import os
import sys

from setuptools import find_packages, setup

ext_modules, cmdclass = [], {}
if os.environ.get("CCK_NO_EXT") != "1":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    arch = os.environ.get("CCK_ARCH", "89")
    nvcc_flags = [
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ]
    # recent torch headers require C++20 for host code; the .cu files are torch-free (C++17 is enough)
    cxx_flags = ["/O2", "/std:c++20"] if sys.platform == "win32" else ["-O3", "-std=c++20"]
    ext_modules.append(
        CUDAExtension(
            name="cckernel._C",
            sources=["csrc/bindings.cpp"] + sorted(glob.glob("csrc/*.cu")),
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        )
    )
    cmdclass = {"build_ext": BuildExtension}

setup(
    name="cckernel",
    version="0.1.0",
    packages=find_packages(include=["cckernel", "cckernel.*"]),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
