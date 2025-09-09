#!/usr/bin/env python3
"""FastAPI server for onnxcrepe inference."""

import asyncio
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging
from io import BytesIO
import os
import stat
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import soundfile as sf
import numpy as np

from onnxcrepe.runner import CrepeRunner

logger = logging.getLogger(__name__)


class ModelServer:
    """Server wrapper for CREPE model."""
    
    def __init__(
        self,
        model: str = "full",
        precision: float = 10.0,
        fmin: float = 50.0,
        fmax: float = 2006.0,
        decoder: str = "weighted_viterbi",
        batch_size: int = 32,
        device_id: int = 0,
        trt_cache: Optional[Path] = None,
        use_trt: bool = True,
        use_cuda: bool = True,
        optimized_model_filepath: Optional[str] = None,
    ):
        """Initialize model server.
        
        Args:
            model: Model capacity
            precision: Time precision in milliseconds
            fmin: Minimum frequency
            fmax: Maximum frequency
            decoder: Decoder type
            batch_size: Batch size
            device_id: GPU device ID
            trt_cache: TensorRT cache directory
            use_trt: Whether to use TensorRT
            use_cuda: Whether to use CUDA
        """
        self.model = model
        self.precision = precision
        self.fmin = fmin
        self.fmax = fmax
        self.decoder = decoder
        self.batch_size = batch_size
        self.device_id = device_id
        self.trt_cache = trt_cache
        self.use_trt = use_trt
        self.use_cuda = use_cuda
        self.optimized_model_filepath = optimized_model_filepath
        
        self.runner = None
        self._init_runner()
    
    def _init_runner(self):
        """Initialize the CREPE runner."""
        logger.info(f"Initializing CREPE model: {self.model}")
        
        self.runner = CrepeRunner(
            model=self.model,
            precision=self.precision,
            fmin=self.fmin,
            fmax=self.fmax,
            decoder=self.decoder,
            batch_size=self.batch_size,
            device_id=self.device_id,
            trt_cache=self.trt_cache,
            use_trt=self.use_trt,
            use_cuda=self.use_cuda,
            optimized_model_filepath=self.optimized_model_filepath,
        )
        
        logger.info("Model runner initialized successfully")
    
    async def process_audio(
        self,
        audio_bytes: bytes,
        return_periodicity: bool = False,
        format: str = "json",
    ) -> Dict[str, Any]:
        """Process audio and return pitch detection results.
        
        Args:
            audio_bytes: Raw audio bytes
            return_periodicity: Whether to return periodicity
            format: Output format
            
        Returns:
            Dictionary with results
        """
        try:
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            logger.info(f"Processing audio file: {tmp_path}")
            
            # Run inference
            result = self.runner.predict_from_file(
                tmp_path,
                return_periodicity=return_periodicity
            )
            
            # Unpack results
            if return_periodicity:
                pitch, periodicity = result
            else:
                pitch = result
                periodicity = None
            
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
            
            # Format results based on requested format
            if format == "json":
                # Handle batch dimension - pitch shape is (1, n_frames)
                if pitch.ndim == 2:
                    pitch_1d = pitch[0]
                else:
                    pitch_1d = pitch
                
                # Create time array
                time = np.arange(len(pitch_1d)) * (self.precision / 1000.0)
                
                response = {
                    "model": self.model,
                    "decoder": self.decoder,
                    "precision_ms": self.precision,
                    "fmin": self.fmin,
                    "fmax": self.fmax,
                    "time": time.tolist(),
                    "pitch": pitch_1d.tolist(),
                }
                
                if periodicity is not None:
                    periodicity_1d = periodicity[0] if periodicity.ndim == 2 else periodicity
                    response["periodicity"] = periodicity_1d.tolist()
                
                return response
            
            elif format == "csv":
                # Return as CSV string
                import pandas as pd
                time = np.arange(len(pitch)) * (self.precision / 1000.0)
                
                data = {"time": time, "pitch": pitch}
                if periodicity is not None:
                    data["periodicity"] = periodicity
                
                df = pd.DataFrame(data)
                csv_string = df.to_csv(index=False)
                
                return {"format": "csv", "data": csv_string}
            
            else:
                # Return numpy arrays as lists
                response = {"pitch": pitch.tolist()}
                if periodicity is not None:
                    response["periodicity"] = periodicity.tolist()
                return response
            
        except Exception as e:
            logger.error(f"Error during audio processing: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


def create_app(model_server: ModelServer) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        model_server: Model server instance
        
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="OnnxCrepe Server",
        version="0.1.0",
        description="Real-time pitch detection with ONNX/TensorRT"
    )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "model": model_server.model,
            "decoder": model_server.decoder,
            "precision_ms": model_server.precision,
        }
    
    @app.post("/predict")
    async def predict(
        audio_file: UploadFile = File(..., description="Audio file to process"),
        return_periodicity: bool = Form(True, description="Return periodicity/confidence"),
        format: str = Form("json", description="Output format (json/csv/raw)"),
    ):
        """Predict pitch from uploaded audio file.
        
        Args:
            audio_file: Audio file to process
            return_periodicity: Whether to return periodicity
            format: Output format
            
        Returns:
            Pitch detection results
        """
        if not audio_file:
            raise HTTPException(status_code=400, detail="No audio file provided")
        
        # Validate format
        if format not in ["json", "csv", "raw"]:
            raise HTTPException(status_code=400, detail=f"Invalid format: {format}")
        
        # Read audio file
        audio_bytes = await audio_file.read()
        
        # Process audio
        results = await model_server.process_audio(
            audio_bytes,
            return_periodicity=return_periodicity,
            format=format
        )
        
        return JSONResponse(content=results)
    
    @app.post("/predict/batch")
    async def predict_batch(
        audio_files: List[UploadFile] = File(..., description="Audio files to process"),
        return_periodicity: bool = Form(True, description="Return periodicity/confidence"),
        format: str = Form("json", description="Output format"),
    ):
        """Process multiple audio files.
        
        Args:
            audio_files: List of audio files
            return_periodicity: Whether to return periodicity
            format: Output format
            
        Returns:
            List of results for each file
        """
        if not audio_files:
            raise HTTPException(status_code=400, detail="No audio files provided")
        
        results = []
        
        for audio_file in audio_files:
            audio_bytes = await audio_file.read()
            
            try:
                result = await model_server.process_audio(
                    audio_bytes,
                    return_periodicity=return_periodicity,
                    format=format
                )
                
                results.append({
                    "filename": audio_file.filename,
                    "status": "success",
                    "result": result
                })
            except Exception as e:
                results.append({
                    "filename": audio_file.filename,
                    "status": "error",
                    "error": str(e)
                })
        
        return JSONResponse(content=results)
    
    @app.get("/info")
    async def info():
        """Get model information."""
        return {
            "model": {
                "capacity": model_server.model,
                "decoder": model_server.decoder,
                "precision_ms": model_server.precision,
                "frequency_range": {
                    "min": model_server.fmin,
                    "max": model_server.fmax,
                },
                "batch_size": model_server.batch_size,
            },
            "hardware": {
                "device_id": model_server.device_id,
                "use_trt": model_server.use_trt,
                "use_cuda": model_server.use_cuda,
            }
        }
    
    return app


async def run_server(
    app: FastAPI,
    socket_path: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
):
    """Run the FastAPI server.
    
    Args:
        app: FastAPI application
        socket_path: UNIX socket path (if None, use HTTP)
        host: Host for HTTP server
        port: Port for HTTP server
    """
    if host or port:
        # Run as HTTP server
        config = uvicorn.Config(
            app,
            host=host or "127.0.0.1",
            port=port or 8000,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()
    else:
        # Run on UNIX socket
        socket_path = socket_path or "/tmp/onnxcrepe.sock"
        
        # Remove existing socket if it exists
        if os.path.exists(socket_path):
            os.unlink(socket_path)
        
        config = uvicorn.Config(
            app,
            uds=socket_path,
            log_level="info",
        )
        server = uvicorn.Server(config)
        
        # Start server and set socket permissions after creation
        async def set_socket_permissions():
            # Wait a moment for the socket to be created
            import asyncio
            await asyncio.sleep(0.1)
            if os.path.exists(socket_path):
                # Set socket permissions (666 for shared access)
                os.chmod(socket_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
        
        # Run permission setting task alongside server
        async with asyncio.TaskGroup() as tg:
            tg.create_task(set_socket_permissions())
            tg.create_task(server.serve())