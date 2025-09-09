#!/usr/bin/env python3
"""Utility functions for crepetrt."""

import os
import hashlib
import logging
import site
from pathlib import Path
import ctypes

logger = logging.getLogger(__name__)


def check_ld_library_path() -> bool:
    """
    Check if LD_LIBRARY_PATH is set to include TensorRT libraries or dynamically load them.
    
    This function follows the same approach as demucs-trt:
    1. First check if system TensorRT libraries are in LD_LIBRARY_PATH
    2. If not, find TensorRT libraries in Python site-packages and load them with ctypes
    """
    
    # Check for system tensorrt libs in the LD_LIBRARY_PATH first
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if ld_library_path and "tensorrt" in ld_library_path.lower():
        logger.info(f"TensorRT libraries found in LD_LIBRARY_PATH: {ld_library_path}")
        return True
    
    # Otherwise we need to dlopen the .so files in the site-packages path so that we can use the tensorrt libs
    try:
        site_packages = Path(site.getsitepackages()[0])
    except (IndexError, AttributeError):
        logger.warning("Could not determine site-packages path")
        return False
    
    # Glob for tensorrt libs in the site-packages path
    tensorrt_libs = list(site_packages.glob("tensorrt*"))
    
    if not tensorrt_libs:
        logger.warning("No TensorRT libraries found in the site-packages path")
        logger.info("TensorRT functionality may be limited. Consider installing tensorrt packages.")
        return False
    
    # Find all .so files in tensorrt directories
    so_files = []
    for path in tensorrt_libs:
        so_files.extend(list(path.glob("*.so*")))
    
    if not so_files:
        logger.warning("No TensorRT .so files found in site-packages")
        return False
    
    # Dynamically load the .so files using ctypes
    loaded_count = 0
    for so_file in so_files:
        try:
            logger.debug(f"Loading TensorRT library: {so_file}")
            ctypes.cdll.LoadLibrary(str(so_file))
            loaded_count += 1
        except OSError as e:
            logger.debug(f"Could not load {so_file}: {e}")
            continue
    
    if loaded_count > 0:
        logger.info(f"Successfully loaded {loaded_count} TensorRT libraries from site-packages")
        return True
    else:
        logger.warning("Failed to load any TensorRT libraries")
        return False


def hash_model_path(model_path: str) -> str:
    """Generate a hash for the model path to use as cache directory name."""
    return hashlib.sha256(model_path.encode()).hexdigest()