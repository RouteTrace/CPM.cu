import torch
import subprocess
import platform
import os

def get_env_info():
    """
    Gathers and returns a string containing system, PyTorch, and CUDA information.
    """
    info = []
    info.append("======== System Information ========")
    info.append(f"OS: {platform.system()} {platform.release()}")
    info.append(f"Python version: {platform.python_version()}")

    info.append("\n======== PyTorch Information ========")
    info.append(f"PyTorch version: {torch.__version__}")
    info.append(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        info.append(f"CUDA version (from PyTorch): {torch.version.cuda}")

    info.append("\n======== CUDA (nvcc) Information ========")
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        nvcc = os.path.join(CUDA_HOME, "bin/nvcc")
        result = subprocess.run([f"{nvcc}", "--version"], capture_output=True, text=True, check=False)
        if result.returncode == 0:
            info.append(result.stdout)
        else:
            info.append("nvcc command found, but returned a non-zero exit code.")
            info.append(f"Stderr: {result.stderr}")

    except FileNotFoundError:
        info.append("nvcc command not found. Is CUDA Toolkit installed and in your PATH?")

    info.append("======== CUDA Devices Information ========")
    info.append(f"GPU capability: {torch.cuda.get_device_capability()}")

    return "\n".join(info)

if __name__ == "__main__":
    print(get_env_info())
