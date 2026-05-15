"""
Utility functions for audio loading, text cleaning, and common operations.
"""

import re
from typing import Optional, Tuple

import torch


def load_audio(
    path: str,
    target_sr: int = 16000,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and resample to target sample rate.
    
    Args:
        path: Path to audio file (WAV, MP3, FLAC, etc.)
        target_sr: Target sample rate (default 16kHz for MMS).
        
    Returns:
        Tuple of (waveform tensor [1, T], sample_rate).
    """
    import torchaudio

    waveform, sr = torchaudio.load(path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
        sr = target_sr

    return waveform, sr


def clean_text(text: str) -> str:
    """
    Basic text normalization: collapse whitespace, strip edges.
    Preserves case and punctuation since they may be relevant.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def romanize_text(text: str) -> str:
    """
    Romanize non-Latin text for MMS forced alignment.
    
    MMS forced alignment expects Latin characters. For non-Latin scripts,
    we use a simple transliteration. For production use, the `uroman`
    tool from Meta is recommended.
    
    This is a lightweight fallback that handles common cases.
    """
    # Check if text is already mostly Latin
    latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
    total_chars = sum(1 for c in text if c.isalpha())

    if total_chars == 0 or latin_chars / total_chars > 0.8:
        return text.lower()

    # For non-Latin text, try to use uroman if available
    try:
        from uroman import uroman
        return uroman(text).lower()
    except ImportError:
        pass

    # Basic fallback: just lowercase and keep what we have
    # MMS tokenizer uses uroman internally for many languages
    return text.lower()


def get_device(preferred: str = "auto") -> torch.device:
    """
    Get the best available device.
    
    Args:
        preferred: "auto", "cuda", "cpu", or specific device string.
        
    Returns:
        torch.device object.
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(preferred)


def estimate_number_audio_region(
    text: str,
    span_start: int,
    span_end: int,
    total_audio_duration: float,
) -> Tuple[float, float]:
    """
    Estimate the time region in audio where a number span is likely spoken.
    
    Uses a simple proportional estimation based on character position.
    This provides a rough region for targeted scoring, reducing computation.
    
    Args:
        text: Full text string.
        span_start: Start character index of the number span.
        span_end: End character index of the number span.
        total_audio_duration: Total audio duration in seconds.
        
    Returns:
        Tuple of (start_time, end_time) in seconds, with padding.
    """
    text_len = max(len(text), 1)

    # Proportional position
    start_ratio = span_start / text_len
    end_ratio = span_end / text_len

    # Convert to time with padding
    padding = 0.5  # seconds of padding on each side
    start_time = max(0, start_ratio * total_audio_duration - padding)
    end_time = min(total_audio_duration, end_ratio * total_audio_duration + padding)

    # Ensure minimum window size
    min_window = 1.0  # seconds
    if end_time - start_time < min_window:
        center = (start_time + end_time) / 2
        start_time = max(0, center - min_window / 2)
        end_time = min(total_audio_duration, center + min_window / 2)

    return start_time, end_time
