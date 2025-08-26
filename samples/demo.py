import numpy as np
import json

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

@profile
def main():
    # Load audio
 


    



    audio_path = Path(__file__).parent / "assets" / "vocadito_12.wav"
   
   # Load audio (keep original sample rate)
    audio, sr = torchaudio_load(audio_path)

    # Convert to mono if not already
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    target_sr = 16000
    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        audio = resampler(audio)
        sr = target_sr

    print(f"audio shape: {audio.shape}")
    print(f"sample rate: {sr}")





    # Here we'll use a 10 millisecond hop length
    precision =10.0

    # Provide a sensible frequency range for your domain (upper limit is 2006 Hz)
    # This would be a reasonable range for speech
    fmin = 50
    fmax = 1100


    # Choose execution providers to use for inference
    providers = [
        'TensorrtExecutionProvider', 
    'CUDAExecutionProvider', 
    'CPUExecutionProvider']

    # Pick a batch size that doesn't cause memory errors on your device
    batch_size = 16384
    # WE need to figure our how to set provider options for CUDAExecutionProvider

    trt_options = {
        "device_id": 0,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache",
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "./trt_cache",
        "trt_fp16_enable": True,
        "trt_max_workspace_size": str(20 * 1024 * 1024 * 1024),  # in bytes, but must be str
    }
    # ort.set_default_logger_severity(0) # Turn on verbose mode for ORT TRT



    cudnn_options = {
        "device_id": 0,
        "cudnn_conv_algo_search": "DEFAULT",
        "cudnn_conv_use_max_workspace": 1,
        "do_copy_in_default_stream": True,
    }

    provider_options =[
        trt_options,
        cudnn_options,  {}]
    so = ort.SessionOptions()
    # so.log_severity_level = 0          # VERBOSE (0=VERBOSE,1=INFO,2=WARNING,3=ERROR,4=FATAL)
    # so.log_verbosity_level = 1         # extra detail

    so.intra_op_num_threads = 0   # let ORT decide
    so.inter_op_num_threads = 0
    # Create inference session
    session = CrepeInferenceSession(
        model='full',
        sess_options=so,
        providers=providers,
        provider_options=provider_options)

    # Compute pitch using the default DirectML GPU or CPU
    # Test with argmax decoder first to see if it works
    pitch, periodicity = onnxcrepe.predict(session, audio, sr, precision=precision, fmin=fmin, fmax=fmax, batch_size=batch_size, return_periodicity=True, decoder=onnxcrepe.decode.weighted_viterbi)


    # Create time axis for plotting
    time_axis = np.arange(pitch.shape[1]) * (precision / 1000.0)  # Convert ms to seconds
    
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
    plt.savefig("pitch_contour_spectrogram.png", dpi=150, bbox_inches='tight')
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
    plt.savefig("periodicity.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save results to JSON
    output_data = {
        "metadata": {
            "audio_file": str(audio_path),
            "sample_rate": int(sr),
            "precision_ms": precision,
            "fmin_hz": fmin,
            "fmax_hz": fmax,
            "batch_size": batch_size,
            "duration_seconds": float(time_axis[-1]),
            "num_frames": int(pitch.shape[1])
        },
        "time_axis": time_axis.tolist(),
        "pitch_hz": pitch[0].tolist(),
        "periodicity": periodicity[0].tolist(),
        "statistics": {
            "pitch_min_hz": float(np.nanmin(pitch[0])),
            "pitch_max_hz": float(np.nanmax(pitch[0])),
            "pitch_mean_hz": float(np.nanmean(pitch[0])),
            "pitch_std_hz": float(np.nanstd(pitch[0])),
            "periodicity_mean": float(np.mean(periodicity[0])),
            "periodicity_std": float(np.std(periodicity[0]))
        }
    }
    
    # Save to JSON file
    output_file = "pitch_analysis_results.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print statistics
    print(f"\n=== Pitch Analysis Results ===")
    print(f"Output shape: {pitch.shape} (batch_size, time_frames)")
    print(f"Duration: {time_axis[-1]:.2f} seconds")
    print(f"Time resolution: {precision} ms")
    print(f"Pitch range: {np.nanmin(pitch[0]):.2f} - {np.nanmax(pitch[0]):.2f} Hz")
    print(f"Mean pitch: {np.nanmean(pitch[0]):.2f} Hz")
    print(f"Mean periodicity: {np.mean(periodicity[0]):.3f}")
    print(f"Results saved to: {output_file}")
    print(f"Spectrogram plot saved to: pitch_contour_spectrogram.png")
    print(f"Periodicity plot saved to: periodicity.png")



if __name__ == "__main__":
    main()