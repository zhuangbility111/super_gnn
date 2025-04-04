from setuptools import setup, find_packages
from torch.utils import cpp_extension
import subprocess
import glob


def check_cuda_support():
    try:
        result = subprocess.run(["nvcc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print("CUDA is supported.")
            return True
        else:
            print("CUDA is not supported.")
            return False
    except FileNotFoundError:
        print("CUDA is not installed.")
        return False


def get_extensions():
    cuda_compile_args = ["-O3", "-DUSE_CUDA", "-x", "cu", "--expt-extended-lambda", "-std=c++17"]
    extra_link_args = []
    cuda_source_files = []

    if check_cuda_support():
        cuda_source_files += glob.glob("csrc/*.cu")
        cuda_source_files.extend(["csrc/ops.cpp"])
    else:
        raise RuntimeError("CUDA is not supported or nvcc is not installed.")

    extensions = [
        cpp_extension.CUDAExtension(
            "supergnn_ops",  # Single module for CUDA
            cuda_source_files,
            include_dirs=["csrc/"],
            extra_compile_args={"nvcc": cuda_compile_args},
            extra_link_args=extra_link_args,
        )
    ]

    return extensions


setup(
    name="supergnn_ops",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    packages=find_packages(),
)
