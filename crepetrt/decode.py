import librosa
import numpy as np
import torch
import torbi

import crepetrt
###############################################################################
# Probability sequence decoding methods
###############################################################################


def argmax(logits):
    """Sample observations by taking the argmax"""
    bins = logits.argmax(axis=1)

    # Convert to frequency in Hz
    return bins, crepetrt.convert.bins_to_frequency(bins)


def weighted_argmax(logits: np.ndarray):
    """Sample observations using weighted sum near the argmax"""
    # Find center of analysis window
    bins = logits.argmax(axis=1)

    return bins, _apply_weights(logits, bins)
def viterbi_torbi(logits, transition, device="cuda"):
    """
    Use torbi library for optimized Viterbi decoding
    logits: [B, T, N] or [T, N] (batch, time, states) 
    transition: [N, N] transition matrix
    """
    # Convert to torch tensor if needed
    if not isinstance(logits, torch.Tensor):
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
    else:
        logits_tensor = logits
    
    # Handle different input shapes
    # torbi expects (batch, frames, states)
    if logits_tensor.dim() == 2:
        # [T, N] -> [1, T, N]
        logits_tensor = logits_tensor.unsqueeze(0)
    elif logits_tensor.dim() == 3:
        # Check if we have [B, N, T] shape (states in middle dimension)
        if logits_tensor.shape[1] == 360 and logits_tensor.shape[2] != 360:
            # [B, N, T] -> [B, T, N]
            logits_tensor = logits_tensor.transpose(1, 2)
    
    # Move to device
    if device == "cuda" and torch.cuda.is_available():
        gpu_idx = 0
        logits_tensor = logits_tensor.to(f"cuda:{gpu_idx}")
        if not isinstance(transition, torch.Tensor):
            transition = torch.tensor(transition, dtype=torch.float32, device=f"cuda:{gpu_idx}")
        else:
            transition = transition.to(f"cuda:{gpu_idx}")
    else:
        gpu_idx = None
        logits_tensor = logits_tensor.cpu()
        if not isinstance(transition, torch.Tensor):
            transition = torch.tensor(transition, dtype=torch.float32)
        else:
            transition = transition.cpu()
    
    # Convert logits to probabilities using softmax
    # Handle -inf values by setting them to a very negative number
    logits_clean = torch.where(torch.isfinite(logits_tensor), logits_tensor, -1e10)
    probabilities = torch.softmax(logits_clean, dim=-1)
    
    # Use torbi for decoding
    # torbi expects shape (batch, frames, states)
    bins = torbi.from_probabilities(
        observation=probabilities,
        transition=transition,
        log_probs=False,  # Already probabilities
        gpu=gpu_idx,
        num_threads=1  # Can increase for CPU processing
    )
    
    return bins


def get_transition(device="cuda"):
    xx, yy = torch.meshgrid(
        torch.arange(360, device=device),
        torch.arange(360, device=device),
        indexing="ij"
    )
    transition = torch.clamp(12 - torch.abs(xx - yy), min=0).float()
    transition /= transition.sum(dim=1, keepdim=True)
    return transition

def viterbi(logits, transition=None, device="cuda"):
    """Sample observations using viterbi decoding with torbi"""
    if transition is None:
        transition = get_transition(device=device)
    
    # Use torbi for optimized Viterbi decoding
    bins = viterbi_torbi(logits, transition, device=device)
    
    # Convert back to numpy
    bins_np = bins.cpu().numpy()
    
    frequencies = crepetrt.convert.bins_to_frequency(bins_np)
    
    return bins_np, frequencies



def weighted_viterbi(logits):
    """Sample observations combining viterbi decoding and weighted argmax"""
    bins, _ = viterbi(logits)

    return bins, _apply_weights(logits, bins)


def _apply_weights(logits, bins):
    # Find bounds of analysis window
    start = np.maximum(0, bins - 4)
    end = np.minimum(logits.shape[1], bins + 5)

    # Mask out everything outside of window
    for batch in range(logits.shape[0]):
        for time in range(logits.shape[2]):
            logits[batch, :start[batch, time], time] = float('-inf')
            logits[batch, end[batch, time]:, time] = float('-inf')

    # Construct weights
    if not hasattr(_apply_weights, 'weights'):
        weights = crepetrt.convert.bins_to_cents(np.arange(360))
        _apply_weights.weights = weights[None, :, None]

    # Convert to probabilities (ReLU)
    probs = np.maximum(0, logits)

    # Apply weights
    prob_sums = probs.sum(axis=1)
    
    # Handle cases where all probabilities are zero (heavily masked)
    zero_mask = prob_sums == 0
    if np.any(zero_mask):
        # Fall back to bin centers for zero probability frames
        bin_cents = crepetrt.convert.bins_to_cents(bins)
        cents = np.where(zero_mask, bin_cents, 
                        (_apply_weights.weights * probs).sum(axis=1) / prob_sums)
    else:
        cents = (_apply_weights.weights * probs).sum(axis=1) / prob_sums

    # Convert to frequency in Hz
    return crepetrt.convert.cents_to_frequency(cents)
