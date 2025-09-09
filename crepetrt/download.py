import os
import threading
from pathlib import Path
from typing import Optional

import requests
import tqdm


BASE_URL = "https://github.com/yqzhishen/crepetrt/releases/download/v1.1.0"
AVAILABLE_MODELS = ["full", "large", "medium", "small", "tiny"]

# Thread lock for safe concurrent downloading
_download_lock = threading.Lock()


def get_cache_dir() -> Path:
    """Get the cache directory for ONNX models.
    
    Returns:
        Path to cache directory (creates if doesn't exist)
    """
    cache_dir = os.environ.get('crepetrt_CACHE_DIR')
    if cache_dir:
        cache_path = Path(cache_dir)
    else:
        # Default to ~/.cache/crepetrt
        cache_path = Path.home() / '.cache' / 'crepetrt'
    
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


def get_model_path(model: str) -> Path:
    """Get the expected path for a model file.
    
    Args:
        model: Model name (e.g., 'full', 'tiny')
        
    Returns:
        Path to model file
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model}'. Available models: {AVAILABLE_MODELS}")
    
    # First check if model exists in legacy assets directory for backward compatibility
    legacy_path = Path(__file__).parent / 'assets' / f'{model}.onnx'
    if legacy_path.exists():
        return legacy_path
    
    # Otherwise use cache directory
    return get_cache_dir() / f'{model}.onnx'


def download_model(model: str, verbose: bool = True) -> Path:
    """Download a model if it doesn't exist.
    
    Args:
        model: Model name to download
        verbose: Whether to show download progress
        
    Returns:
        Path to downloaded model file
        
    Raises:
        ValueError: If model name is invalid
        requests.RequestException: If download fails
    """
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{model}'. Available models: {AVAILABLE_MODELS}")
    
    model_path = get_model_path(model)
    
    # Check if model already exists
    if model_path.exists():
        return model_path
    
    # Thread-safe download check
    with _download_lock:
        # Double-check after acquiring lock
        if model_path.exists():
            return model_path
        
        if verbose:
            print(f"Downloading ONNX CREPE model '{model}' to {model_path}")
        
        # Download model
        url = f"{BASE_URL}/{model}.onnx"
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file size for progress bar
            total_size = int(response.headers.get('content-length', 0))
            
            # Create parent directory if needed
            model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            with open(model_path, 'wb') as f:
                if verbose and total_size > 0:
                    with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {model}.onnx") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                else:
                    # No progress bar if size unknown or verbose=False
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            if verbose:
                print(f"Successfully downloaded {model}.onnx")
            
        except requests.RequestException as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            raise requests.RequestException(f"Failed to download model '{model}': {e}")
    
    return model_path


def ensure_model_available(model: str, verbose: bool = True) -> Path:
    """Ensure a model is available, downloading if necessary.
    
    Args:
        model: Model name to ensure is available
        verbose: Whether to show download progress
        
    Returns:
        Path to model file
        
    Raises:
        ValueError: If model name is invalid
        requests.RequestException: If download fails
        FileNotFoundError: If model file doesn't exist after download
    """
    model_path = download_model(model, verbose)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found after download: {model_path}")
    
    return model_path


def download_all_models(verbose: bool = True) -> dict:
    """Download all available models.
    
    Args:
        verbose: Whether to show download progress
        
    Returns:
        Dict mapping model names to their paths
    """
    model_paths = {}
    
    for model in AVAILABLE_MODELS:
        try:
            model_paths[model] = download_model(model, verbose)
        except Exception as e:
            if verbose:
                print(f"Failed to download {model}: {e}")
            continue
    
    return model_paths