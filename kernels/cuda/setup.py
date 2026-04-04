"""Build script for DEF-attackvla CUDA kernels."""
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="attackvla_cuda_kernels",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            "attackvla_cuda_kernels",
            ["patch_guard_kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-gencode=arch=compute_89,code=sm_89",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
