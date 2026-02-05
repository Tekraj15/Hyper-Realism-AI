import platform
import torch
from .engine_apple_silicon import AppleSiliconEngine
from .engine_nvidia_cuda import NvidiaCUDAEngine

def get_optimized_engine(config):
    """
    Detects hardware and returns the specifically optimized engine instance.
    """
    system = platform.system()
    processor = platform.processor()
    
    # Precise detection for Apple Silicon (M1/M2/M3)
    is_apple_silicon = (system == 'Darwin' and 'arm' in processor.lower())
    
    if is_apple_silicon:
        print(" !! Detected Apple Silicon (M-Series) Environment.")
        print(" Loading 'AppleSiliconEngine' with Unified Memory Optimization...")
        return AppleSiliconEngine(config)
    
    # Precise detection for NVIDIA CUDA
    elif torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f" !! Detected NVIDIA GPU ({gpu_name}).")
        print(" Loading 'NvidiaCUDAEngine' with ZeroGPU Optimization...")
        return NvidiaCUDAEngine(config)
    
    else:
        raise RuntimeError(" Fatal: No compatible AI Accelerator (NVIDIA CUDA or Apple Silicon) found.")