"""
speech_number_norm — Multilingual Speech Number Normalization

Given a (speech, text) pair, converts numbers in the text to the form
spoken in the audio. Uses candidate generation (num2words) + CTC forced
alignment scoring (MMS) to select the best verbalization.
"""

from speech_number_norm.normalizer import SpeechNumberNormalizer
from speech_number_norm.detector import NumberDetector, NumberSpan
from speech_number_norm.candidates import CandidateGenerator
from speech_number_norm.scorer import CTCScorer

__version__ = "0.1.0"

__all__ = [
    "SpeechNumberNormalizer",
    "NumberDetector",
    "NumberSpan",
    "CandidateGenerator",
    "CTCScorer",
]
