#!/usr/bin/env python3
"""Main CLI interface for onnxcrepe using Typer."""

from pathlib import Path
from typing import Optional, List
import asyncio
import json
import sys
import logging

import typer
import onnxruntime as ort
from coloredlogs import install

from onnxcrepe.runner import CrepeRunner
from onnxcrepe.utils import check_ld_library_path, hash_model_path
import onnxcrepe
import line_profiler
# Configure logging to always use stderr
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure coloredlogs to use stderr
install(level="INFO", stream=sys.stderr)

app = typer.Typer(help="OnnxCrepe: Real-time pitch detection with ONNX/TensorRT")


@app.command("predict")
@line_profiler.profile
def predict(
    audio_files: List[str] = typer.Argument(..., help="Audio files to process"),
    output_dir: Optional[Path] = typer.Option(None, "-o", "--output-dir", help="Output directory"),
    model: str = typer.Option("full", "-m", "--model", help="Model capacity (full/large/medium/small/tiny)"),
    precision: float = typer.Option(10.0, "--precision", help="Time precision in milliseconds"),
    fmin: float = typer.Option(50.0, "--fmin", help="Minimum frequency in Hz"),
    fmax: float = typer.Option(onnxcrepe.MAX_FMAX, "--fmax", help="Maximum frequency in Hz"),
    decoder: str = typer.Option("weighted_viterbi", "-d", "--decoder", help="Decoder type"),
    batch_size: int = typer.Option(32, "-b", "--batch", help="Batch size"),
    pad: bool = typer.Option(True, "--pad/--no-pad", help="Zero-pad audio"),
    save_periodicity: bool = typer.Option(True, "--save-periodicity/--no-save-periodicity", help="Save periodicity/confidence"),
    format: str = typer.Option("csv", "-f", "--format", help="Output format (csv/npy/json)"),
    no_trt: bool = typer.Option(False, "--no-trt", help="Disable TensorRT"),
    no_cuda: bool = typer.Option(False, "--no-cuda", help="Disable CUDA"),
    trt_cache: Path = typer.Option(
        Path.home() / ".cache/onnxcrepe/trt_engine_cache",
        "--trt-cache",
        help="TensorRT cache directory"
    ),
    device_id: int = typer.Option(0, "--device-id", help="GPU device ID"),
    optimized_model_filepath: Optional[str] = typer.Option(None, "--optimized-model-filepath", help="Path to optimized model"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose output"),
    n_jobs: int = typer.Option(1, "-j", "--jobs", help="Number of parallel jobs"),
):
    """Predict pitch from audio files."""
    
    # Suppress info logs when outputting JSON to keep stdout clean
    if format == "json" and not verbose:
        logging.getLogger().setLevel(logging.WARNING)
        install(level="WARNING", stream=sys.stderr)
    
    if verbose:
        ort.set_default_logger_severity(0)
        ort.set_default_logger_verbosity(0)
    
    # Validate inputs
    if format not in ["csv", "npy", "json"]:
        logger.error(f"Invalid format: {format}. Must be csv, npy, or json")
        raise typer.Exit(1)
    
    if model not in ["full", "large", "medium", "small", "tiny"]:
        logger.error(f"Invalid model: {model}. Must be full, large, medium, small, or tiny")
        raise typer.Exit(1)
    
    # Check audio files exist
    audio_paths = []
    for audio_file in audio_files:
        path = Path(audio_file)
        if not path.exists():
            logger.error(f"Audio file not found: {audio_file}")
            raise typer.Exit(1)
        audio_paths.append(path)
    
    # Check LD_LIBRARY_PATH if using GPU
    if not no_trt or not no_cuda:
        if not check_ld_library_path():
            logger.warning("TensorRT libraries may not be available")
            logger.info("Will attempt to continue - may fall back to CUDA or CPU")
    
    # Setup TRT cache
    if not no_trt:
        model_hash = hash_model_path(model)
        model_cache = trt_cache / model_hash
        
        if not model_cache.exists():
            logger.warning(f"TensorRT cache directory {model_cache} does not exist.")
            logger.warning("Will need to build the engine from scratch.")
            logger.warning("This will take a while and require significant GPU memory.")
            
            if not typer.confirm("Do you want to continue?"):
                logger.error("Aborting...")
                raise typer.Exit(1)
            
            logger.info(f"Creating TensorRT cache directory: {model_cache}")
            model_cache.mkdir(parents=True, exist_ok=True)
    else:
        model_cache = None
    
    # Log configuration
    logger.info(f"Model: {model}")
    logger.info(f"Decoder: {decoder}")
    logger.info(f"Precision: {precision}ms")
    logger.info(f"Frequency range: {fmin}-{fmax} Hz")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Output format: {format}")
    if output_dir:
        logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize runner
        runner = CrepeRunner(
            model=model,
            precision=precision,
            fmin=fmin,
            fmax=fmax,
            decoder=decoder,
            batch_size=batch_size,
            pad=pad,
            device_id=device_id,
            trt_cache=model_cache,
            use_trt=not no_trt,
            use_cuda=not no_cuda,
            optimized_model_filepath=optimized_model_filepath,
        )
        
        # Process files
        results = runner.process_files(
            audio_paths,
            output_dir=output_dir,
            format=format,
            save_periodicity=save_periodicity,
            n_jobs=n_jobs,
        )
        
        # Output results
        if format == "json" and len(audio_paths) == 1:
            # For single file JSON, output to stdout
            json_output = {}
            for key, paths in results.items():
                json_output[key] = str(paths[0]) if paths else None
            print(json.dumps(json_output))
        else:
            # Log saved files
            for key, paths in results.items():
                for path in paths:
                    logger.info(f"Saved {key}: {path}")
        
        logger.info("Processing complete!")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise typer.Exit(1)


@app.command("serve")
def serve(
    model: str = typer.Option("full", "-m", "--model", help="Model capacity"),
    socket_path: str = typer.Option("/tmp/onnxcrepe.sock", "-s", "--socket", help="UNIX socket path"),
    host: Optional[str] = typer.Option(None, "-h", "--host", help="Host to bind to (overrides socket)"),
    port: Optional[int] = typer.Option(None, "-p", "--port", help="Port to bind to (overrides socket)"),
    precision: float = typer.Option(10.0, "--precision", help="Time precision in milliseconds"),
    fmin: float = typer.Option(50.0, "--fmin", help="Minimum frequency in Hz"),
    fmax: float = typer.Option(onnxcrepe.MAX_FMAX, "--fmax", help="Maximum frequency in Hz"),
    decoder: str = typer.Option("weighted_viterbi", "-d", "--decoder", help="Decoder type"),
    batch_size: int = typer.Option(32, "-b", "--batch", help="Batch size"),
    no_trt: bool = typer.Option(False, "--no-trt", help="Disable TensorRT"),
    no_cuda: bool = typer.Option(False, "--no-cuda", help="Disable CUDA"),
    trt_cache: Path = typer.Option(
        Path.home() / ".cache/onnxcrepe/trt_engine_cache",
        "--trt-cache",
        help="TensorRT cache directory"
    ),
    device_id: int = typer.Option(0, "--device-id", help="GPU device ID"),
    optimized_model_filepath: Optional[str] = typer.Option(None, "--optimized-model-filepath", help="Path to optimized model"),
):
    """Serve the model over HTTP or UNIX socket."""
    from onnxcrepe.server import ModelServer, create_app, run_server
    
    # Check LD_LIBRARY_PATH
    if not no_trt or not no_cuda:
        if not check_ld_library_path():
            logger.warning("TensorRT libraries may not be available")
            logger.info("Will attempt to continue - may fall back to CUDA or CPU")
    
    # Setup TRT cache
    if not no_trt:
        model_hash = hash_model_path(model)
        model_cache = trt_cache / model_hash
        
        if not model_cache.exists():
            logger.warning(f"TensorRT cache directory {model_cache} does not exist.")
            logger.warning("Will need to build the engine from scratch.")
            
            if not typer.confirm("Do you want to continue?"):
                raise typer.Exit(1)
            
            model_cache.mkdir(parents=True, exist_ok=True)
    else:
        model_cache = None
    
    logger.info(f"Starting server with model: {model}")
    logger.info(f"Decoder: {decoder}")
    logger.info(f"Precision: {precision}ms")
    
    # Create model server
    model_server = ModelServer(
        model=model,
        precision=precision,
        fmin=fmin,
        fmax=fmax,
        decoder=decoder,
        batch_size=batch_size,
        device_id=device_id,
        trt_cache=model_cache,
        use_trt=not no_trt,
        use_cuda=not no_cuda,
        optimized_model_filepath=optimized_model_filepath,
    )
    
    # Create FastAPI app
    app_instance = create_app(model_server)
    
    # Run server
    if host or port:
        logger.info(f"Starting HTTP server on {host or '127.0.0.1'}:{port or 8000}")
    else:
        logger.info(f"Starting server on UNIX socket: {socket_path}")
    
    asyncio.run(run_server(app_instance, socket_path, host, port))


if __name__ == "__main__":
    app()