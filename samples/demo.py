import numpy as np
import json
import time

import onnxcrepe
from onnxcrepe.session import CrepeInferenceSession
from pathlib import Path
import onnxruntime as ort
from line_profiler import profile

from torchaudio import load as torchaudio_load
import torchaudio.transforms as T
from torch import inference_mode
import matplotlib.pyplot as plt
import librosa
import librosa.display
from joblib import Memory
import argparse


def build_session(device_id=0):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 0
    so.inter_op_num_threads = 0
    providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    trt_options = {
    "device_id": device_id,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "./trt_cache",   # folder to store compiled TRT engines
    "trt_timing_cache_enable": True,
    "trt_timing_cache_path": "./trt_cache",   # timing info speeds up future builds
    "trt_fp16_enable": True,
   
    }
    cudnn_options = {"device_id": device_id}
    return onnxcrepe.CrepeInferenceSession(
        model="full", sess_options=so, providers=providers, provider_options=[trt_options, cudnn_options, {}]
    )

def plot_results(pitch, periodicity, audio, sr, precision, fmin, fmax, output_path):
         # Create time axis for plotting
        time_axis = np.arange(pitch.shape[1]) * (args.precision / 1000.0)  # Convert ms to seconds
        
        # Convert audio to numpy for librosa (from torch tensor)
        audio_np = audio.squeeze().numpy()
        
        # Compute spectrogram using librosa
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_np)), ref=np.max)
        
        # Create spectrogram with pitch contour overlay
        fig, ax = plt.subplots(figsize=(12, 6))
        img = librosa.display.specshow(D, x_axis='time', y_axis='log', sr=sr, ax=ax)
        ax.set_title('Pitch Contour over Spectrogram', fontsize=14)
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        
        # Overlay pitch contour (mask zeros/NaN values for cleaner visualization)
        valid_pitch = pitch[0].copy()
        valid_pitch[valid_pitch == 0] = np.nan  # Replace zeros with NaN
        ax.plot(time_axis, valid_pitch, label='F0', color='cyan', linewidth=3)
        ax.legend(loc='upper right')
        ax.set_ylim([fmin, fmax])  # Set y-limit to frequency range
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/pitch_contour_spectrogram.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create separate periodicity plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(time_axis, periodicity[0], linewidth=1.5, color='green')
        ax.set_ylabel('Periodicity/Confidence', fontsize=12)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_title('Periodicity (Voicing Confidence)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])  # Periodicity is typically 0-1
        
        plt.tight_layout()
        plt.savefig(f"{output_path}/periodicity.png", dpi=150, bbox_inches='tight')
        plt.close()
        return 
def preprocess_audio(audio, sr):
       # Convert to mono if not already
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    target_sr = 16000
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
        sr = target_sr
    return audio, sr


@profile
def main():
    # Load audio
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="samples/assets/vocadito_12.wav")
    parser.add_argument("--output_path", type=str, default="samples/outputs")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--precision", type=float, default=10.0)
    parser.add_argument("--fmin", type=float, default=50)
    parser.add_argument("--fmax", type=float, default=1100)
    parser.add_argument("--plot", action="store_true")
    
    args = parser.parse_args()
   
   # Load audio (keep original sample rate)
    audio, sr = torchaudio_load(args.audio_path)
    audio, sr = preprocess_audio(audio, sr)
    
    # Calculate audio duration
    audio_duration = audio.shape[-1] / sr
    
    # Create inference session
    session = build_session(device_id=args.device_id)
    
    # Time the prediction
    start_time = time.time()
    pitch, periodicity = onnxcrepe.predict(session, audio, sr, precision=args.precision, fmin=args.fmin, fmax=args.fmax, batch_size=args.batch_size, return_periodicity=True, decoder=onnxcrepe.decode.weighted_viterbi)
    inference_time = time.time() - start_time
    
    # Calculate realtime factor (how many times faster than realtime)
    realtime_factor = audio_duration / inference_time

    if args.plot:
        plot_results(pitch, periodicity, audio, sr, args.precision, args.fmin, args.fmax, args.output_path)
   
   
    # Print statistics
    print(f"\n=== Pitch Analysis Results ===")
    print(f"Output shape: {pitch.shape} (batch_size, time_frames)")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"Inference time: {inference_time:.3f} seconds")
    print(f"Realtime factor: {realtime_factor:.1f}x realtime (higher is faster)")
    print(f"Time resolution: {args.precision} ms")
    print(f"Pitch range: {np.nanmin(pitch[0]):.2f} - {np.nanmax(pitch[0]):.2f} Hz")
    print(f"Mean pitch: {np.nanmean(pitch[0]):.2f} Hz")
    print(f"Mean periodicity: {np.mean(periodicity[0]):.3f}")




if __name__ == "__main__":
    
    main()