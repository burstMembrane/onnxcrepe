import os
from pathlib import Path

import onnxruntime as ort

from . import download


def find_optimized_model(model_name, optimized_dir="./optimized_models"):
    """Find optimized ONNX model if available."""
    optimized_dir = Path(optimized_dir)
    optimized_path = optimized_dir / f"{model_name}_optimized.onnx"
    return optimized_path if optimized_path.exists() else None


def configure_tensorrt_for_prebuilt_engines(provider_options, cache_path="./trt_engines"):
    """Configure TensorRT execution provider to use pre-built engines."""
    if not provider_options:
        return provider_options
    
    cache_path = Path(cache_path).resolve()
    
    # Find TensorRT options in provider_options
    for i, options in enumerate(provider_options):
        if isinstance(options, dict) and any(key.startswith('trt_') for key in options.keys()):
            # Update TensorRT options for pre-built engine usage
            options.setdefault("trt_engine_cache_enable", True)
            options.setdefault("trt_engine_cache_path", str(cache_path))
            options.setdefault("trt_timing_cache_enable", True)
            options.setdefault("trt_timing_cache_path", str(cache_path))
            
            # Configure for pre-built engine loading
            if cache_path.exists():
                options.setdefault("trt_ep_context_file_path", str(cache_path))
                options.setdefault("trt_ep_context_embed_mode", 0)
            
            break
    
    return provider_options


def optimize_session_options_for_cold_start(sess_options=None):
    """Optimize session options for cold-start performance."""
    if sess_options is None:
        sess_options = ort.SessionOptions()
    
    # Minimize threading overhead for single inference
    sess_options.intra_op_num_threads = 0  # Let ORT decide based on hardware
    sess_options.inter_op_num_threads = 0  # Let ORT decide
    
    # Enable memory optimizations
    sess_options.enable_mem_pattern = True
    sess_options.enable_mem_reuse = True
    
    return sess_options


class CrepeInferenceSession(ort.InferenceSession):
    def __init__(self, model='full', sess_options=None, providers=None, provider_options=None, 
                 use_optimized_model=True, engine_cache_path="./trt_engines", model_path=None, **kwargs):
        """
        Create CREPE inference session with cold-start optimizations.
        
        Args:
            model: Model name ('full', 'large', 'medium', 'small', 'tiny')
            sess_options: ONNX Runtime session options
            providers: List of execution providers
            provider_options: Options for execution providers
            use_optimized_model: Whether to use pre-optimized models if available
            engine_cache_path: Path to TensorRT engine cache directory
            **kwargs: Additional arguments passed to onnxruntime.InferenceSession
        """
        # Preload DLLs for faster initialization
        try:
            ort.preload_dlls()
        except (AttributeError, Exception):
            # preload_dlls might not be available in all versions
            pass
        
        if model_path is None:
           
            
            
            # 1. Try to use pre-optimized model if available and requested
            if use_optimized_model:
                optimized_path = find_optimized_model(model)
                if optimized_path:
                    model_path = optimized_path
                    # Disable graph optimizations since model is already optimized
                    if sess_options is None:
                        sess_options = ort.SessionOptions()
                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            
            # 2. Fall back to original model
            if model_path is None:
                model_path = download.ensure_model_available(model, verbose=True)
        else:
            model_path = model_path
            
        # Optimize session options for cold-start
        sess_options = optimize_session_options_for_cold_start(sess_options)
        
        # Configure TensorRT for pre-built engines if using TensorRT
        if providers and "TensorrtExecutionProvider" in providers:
            provider_options = configure_tensorrt_for_prebuilt_engines(provider_options, engine_cache_path)
        
        # Initialize the session
        super().__init__(str(model_path), sess_options, providers, provider_options, **kwargs)
