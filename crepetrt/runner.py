#!/usr/bin/env python3
"""Unified inference runner for crepetrt."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union
import json

import numpy as np
import soundfile as sf
from tqdm import tqdm
import logging
from joblib import Parallel, delayed

import crepetrt
from crepetrt.session import CrepeInferenceSession
from crepetrt.utils import hash_model_path, check_ld_library_path

logger = logging.getLogger(__name__)


class CrepeRunner:
    """Unified runner for CREPE pitch detection with ONNX/TensorRT."""
    
    def __init__(
        self,
        model: str = "full",
        providers: Optional[list] = None,
        precision: float = 10.0,
        fmin: float = 50.0,
        fmax: float = crepetrt.MAX_FMAX,
        decoder: str = "weighted_viterbi",
        batch_size: int = 32,
        pad: bool = True,
        device_id: int = 0,
        trt_cache: Optional[Path] = None,
        use_trt: bool = True,
        use_cuda: bool = True,
        optimized_model_filepath: Optional[str] = None,
    ):
        """Initialize the CREPE runner.
        
        Args:
            model: Model capacity ('full', 'large', 'medium', 'small', 'tiny')
            providers: ONNX Runtime providers (if None, will be auto-configured)
            precision: Time precision in milliseconds
            fmin: Minimum frequency in Hz
            fmax: Maximum frequency in Hz
            decoder: Decoder type ('argmax', 'weighted_argmax', 'viterbi', 'weighted_viterbi')
            batch_size: Batch size for inference
            pad: Whether to zero-pad audio
            device_id: GPU device ID
            trt_cache: TensorRT cache directory
            use_trt: Whether to use TensorRT
            use_cuda: Whether to use CUDA
            optimized_model_filepath: Path to optimized model
        """
        self.model = model
        self.precision = precision
        self.fmin = fmin
        self.fmax = fmax
        self.batch_size = batch_size
        self.pad = pad
        self.device_id = device_id
        self.use_trt = use_trt
        self.use_cuda = use_cuda
        self.optimized_model_filepath = optimized_model_filepath
        
        # Get decoder function
        self.decoder = self._get_decoder(decoder)
        self.decoder_name = decoder
        
        # Configure providers if not provided
        if providers is None:
            providers = self._configure_providers(trt_cache)
        self.providers = providers
        
        # Initialize session
        self.session = None
        self._init_session()
    
    def _get_decoder(self, decoder_name: str):
        """Get decoder function by name."""
        decoders = {
            'argmax': crepetrt.decode.argmax,
            'weighted_argmax': crepetrt.decode.weighted_argmax,
            'viterbi': crepetrt.decode.viterbi,
            'weighted_viterbi': crepetrt.decode.weighted_viterbi,
        }
        
        if decoder_name not in decoders:
            raise ValueError(f"Unknown decoder: {decoder_name}. "
                           f"Must be one of: {list(decoders.keys())}")
        
        return decoders[decoder_name]
    
    def _configure_providers(self, trt_cache: Optional[Path] = None) -> list:
        """Configure ONNX Runtime providers."""
        import onnxruntime as ort
        
        providers = []
        
        # TensorRT configuration
        if self.use_trt and 'TensorrtExecutionProvider' in ort.get_available_providers():
            trt_options = {
                "trt_engine_cache_enable": True,
                "trt_timing_cache_enable": True,
                "trt_fp16_enable": True,
                "trt_builder_optimization_level": 3,
                "trt_max_workspace_size": "20GB",
                "trt_ep_context_embed_mode": 0,
                "device_id": self.device_id,
            }
            
            # Configure cache paths if provided
            if trt_cache:
                model_hash = hash_model_path(self.model)
                cache_dir = trt_cache / model_hash
                cache_dir.mkdir(parents=True, exist_ok=True)
                
                trt_options["trt_engine_cache_path"] = str(cache_dir)
                trt_options["trt_timing_cache_path"] = str(cache_dir)
                trt_options["trt_ep_context_file_path"] = str(cache_dir)
                
                logger.info(f"Using TensorRT cache: {cache_dir}")
            
            providers.append(("TensorrtExecutionProvider", trt_options))
        
        # CUDA configuration
        if self.use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            cuda_options = {
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 20 * 1024**3,
                "do_copy_in_default_stream": True,
                "cudnn_conv_use_max_workspace": True,
                "device_id": self.device_id,
            }
            providers.append(("CUDAExecutionProvider", cuda_options))
        
        # Fallback to CPU
        if not providers:
            providers.append("CPUExecutionProvider")
            logger.warning("No GPU providers available, using CPU")
        
        return providers
    
    def _init_session(self):
        """Initialize the ONNX Runtime session."""
        import onnxruntime as ort
        
        # Create session options
        options = ort.SessionOptions()
        
        # Special handling for DirectML
        if self.providers and len(self.providers) > 0:
            first_provider = self.providers[0]
            if isinstance(first_provider, tuple):
                provider_name = first_provider[0]
            else:
                provider_name = first_provider
            
            if provider_name == 'DmlExecutionProvider':
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Create inference session
        logger.info(f"Initializing CREPE model: {self.model}")
        logger.info(f"Using providers: {self.providers}")
        
        self.session = CrepeInferenceSession(
            model=self.model,
            sess_options=options,
            providers=self.providers
        )
        
        logger.info("Model initialized successfully")
    
    def predict(
        self,
        audio: Union[np.ndarray, str, Path],
        sample_rate: Optional[int] = None,
        return_periodicity: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict pitch from audio.
        
        Args:
            audio: Audio array or path to audio file
            sample_rate: Sample rate (required if audio is array)
            return_periodicity: Whether to return periodicity/confidence
            
        Returns:
            pitch: Pitch values in Hz
            periodicity: (optional) Periodicity/confidence values
        """
        # Load audio if path provided
        if isinstance(audio, (str, Path)):
            audio, sample_rate = sf.read(str(audio), always_2d=False)
        elif sample_rate is None:
            raise ValueError("sample_rate must be provided when audio is an array")
        
        # Ensure audio is mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Predict
        result = crepetrt.predict(
            self.session,
            audio,
            sample_rate,
            precision=self.precision,
            fmin=self.fmin,
            fmax=self.fmax,
            decoder=self.decoder,
            return_periodicity=return_periodicity,
            batch_size=self.batch_size,
            pad=self.pad
        )
        
        # Return results with batch dimension for compatibility
        # Shape will be (1, n_frames)
        return result
    
    def predict_from_file(
        self,
        audio_path: Union[str, Path],
        return_periodicity: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Predict pitch from audio file.
        
        Args:
            audio_path: Path to audio file
            return_periodicity: Whether to return periodicity
            
        Returns:
            pitch: Pitch values in Hz
            periodicity: (optional) Periodicity values
        """
        return self.predict(audio_path, return_periodicity=return_periodicity)
    
    def save_results(
        self,
        pitch: np.ndarray,
        periodicity: Optional[np.ndarray],
        output_path: Union[str, Path],
        format: str = "csv",
        audio_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Path]:
        """Save pitch detection results to file.
        
        Args:
            pitch: Pitch values
            periodicity: Periodicity values (optional)
            output_path: Output file path or directory
            format: Output format ('csv', 'npy', 'json')
            audio_path: Original audio path (for naming)
            
        Returns:
            Dictionary of saved file paths
        """
        output_path = Path(output_path)
        
        # Determine output file path
        if output_path.is_dir():
            if audio_path:
                base_name = Path(audio_path).stem
            else:
                base_name = "output"
            
            if format == "csv":
                file_path = output_path / f"{base_name}_pitch.csv"
            elif format == "npy":
                file_path = output_path / f"{base_name}_pitch.npy"
            elif format == "json":
                file_path = output_path / f"{base_name}_pitch.json"
            else:
                raise ValueError(f"Unknown format: {format}")
        else:
            file_path = output_path
            # Infer format from extension if not specified
            if format == "csv" and not str(file_path).endswith('.csv'):
                format = file_path.suffix[1:] if file_path.suffix else "csv"
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        saved_paths = {}
        
        if format == "csv":
            import pandas as pd
            
            # Handle batch dimension - pitch shape is (1, n_frames)
            if pitch.ndim == 2:
                pitch_1d = pitch[0]
            else:
                pitch_1d = pitch
            
            # Create time array
            time = np.arange(len(pitch_1d)) * (self.precision / 1000.0)
            
            # Create dataframe
            data = {"time": time, "pitch": pitch_1d}
            if periodicity is not None:
                periodicity_1d = periodicity[0] if periodicity.ndim == 2 else periodicity
                data["periodicity"] = periodicity_1d
            
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
            saved_paths["pitch"] = file_path
            
        elif format == "npy":
            # Save pitch
            np.save(file_path, pitch)
            saved_paths["pitch"] = file_path
            
            # Save periodicity if provided
            if periodicity is not None:
                periodicity_path = file_path.parent / f"{file_path.stem}_periodicity.npy"
                np.save(periodicity_path, periodicity)
                saved_paths["periodicity"] = periodicity_path
                
        elif format == "json":
            # Handle batch dimension - pitch shape is (1, n_frames)
            if pitch.ndim == 2:
                pitch_1d = pitch[0]
            else:
                pitch_1d = pitch
            
            # Create time array
            time = np.arange(len(pitch_1d)) * (self.precision / 1000.0)
            
            # Prepare data
            data = {
                "model": self.model,
                "decoder": self.decoder_name,
                "precision_ms": self.precision,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "time": time.tolist(),
                "pitch": pitch_1d.tolist(),
            }
            
            if periodicity is not None:
                periodicity_1d = periodicity[0] if periodicity.ndim == 2 else periodicity
                data["periodicity"] = periodicity_1d.tolist()
            
            # Save JSON
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            saved_paths["pitch"] = file_path
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return saved_paths
    
    def process_files(
        self,
        audio_files: list[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        format: str = "csv",
        save_periodicity: bool = False,
        n_jobs: int = 1,
    ) -> Dict[str, list[Path]]:
        """Process multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            output_dir: Output directory (if None, saves next to audio files)
            format: Output format
            save_periodicity: Whether to save periodicity
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary mapping result types to file paths
        """
        def process_single(audio_path):
            audio_path = Path(audio_path)
            
            # Determine output directory
            if output_dir:
                out_dir = Path(output_dir)
            else:
                out_dir = audio_path.parent
            
            # Predict
            result = self.predict_from_file(
                audio_path,
                return_periodicity=save_periodicity
            )
            
            # Unpack results - handle batch dimension
            if save_periodicity:
                pitch, periodicity = result
            else:
                pitch = result
                periodicity = None
            
            # Save results
            saved = self.save_results(
                pitch,
                periodicity,
                out_dir,
                format=format,
                audio_path=audio_path
            )
            
            return saved
        
        # Process files
        if n_jobs == 1:
            # Sequential processing with progress bar
            results = []
            for audio_file in tqdm(audio_files, desc="Processing files"):
                results.append(process_single(audio_file))
        else:
            # Parallel processing
            results = Parallel(n_jobs=n_jobs)(
                delayed(process_single)(audio_file)
                for audio_file in tqdm(audio_files, desc="Processing files")
            )
        
        # Aggregate results
        all_saved = {"pitch": [], "periodicity": []}
        for saved in results:
            for key, path in saved.items():
                if key in all_saved:
                    all_saved[key].append(path)
        
        # Remove empty lists
        all_saved = {k: v for k, v in all_saved.items() if v}
        
        return all_saved