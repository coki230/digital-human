# src/avatar/rocm_utils.py
import torch
from loguru import logger


def get_device(prefer: str = "cuda") -> torch.device:
    """统一获取设备，兼容 ROCm（HIP）"""
    if prefer == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
        return device
    logger.warning("CUDA/ROCm not available, falling back to CPU")
    return torch.device("cpu")


def synchronize():
    """兼容 ROCm 的同步"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()