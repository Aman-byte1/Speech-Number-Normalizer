import logging
from typing import Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


class CTCScorer:
    """Scores candidate verbalizations against audio using CTC forced alignment."""

    def __init__(self, device: str = "auto"):
        from speech_number_norm.utils import get_device
        self.device = get_device(device)
        self._model = None
        self._tokenizer = None
        self._aligner = None
        self._sample_rate = 16000
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return

        import torchaudio

        logger.info("Loading MMS forced alignment model...")
        try:
            bundle = torchaudio.pipelines.MMS_FA
            self._model = bundle.get_model().to(self.device)
            self._model.eval()
            self._tokenizer = bundle.get_tokenizer()
            self._aligner = bundle.get_aligner()
            self._sample_rate = bundle.sample_rate
            self._loaded = True
            logger.info(
                f"MMS model loaded on {self.device} "
                f"(sample_rate={self._sample_rate})"
            )
        except Exception as e:
            logger.warning(f"Failed to load MMS model: {e}. Falling back to heuristic scoring.")
            self._loaded = False

    @torch.inference_mode()
    def get_emissions(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract CTC emission probabilities from audio."""
        self._ensure_loaded()
        if not self._loaded:
            return torch.tensor([])

        waveform = waveform.to(self.device)
        emissions, _ = self._model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)
        return emissions

    def score_candidates(
        self,
        waveform: torch.Tensor,
        candidates: List[str],
        language: str = "en",
    ) -> List[Tuple[str, float]]:
        """Score each candidate against the audio and return ranked results."""
        self._ensure_loaded()

        if not self._loaded or len(candidates) == 0:
            # Heuristic fallback: prefer first candidate (most common form)
            return [(c, -i) for i, c in enumerate(candidates)]

        # Get emissions once for the audio
        emissions = self.get_emissions(waveform)
        if emissions.numel() == 0:
            return [(c, -i) for i, c in enumerate(candidates)]

        results = []
        for candidate in candidates:
            score = self._score_single(emissions, candidate)
            results.append((candidate, score))

        # Sort by score descending (higher = better)
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _score_single(
        self,
        emissions: torch.Tensor,
        text: str,
    ) -> float:
        """Compute the forced alignment score for a single candidate."""
        import torchaudio

        try:
            # Romanize text for MMS (expects Latin script)
            from speech_number_norm.utils import romanize_text
            clean_text = romanize_text(text)

            if not clean_text.strip():
                return float("-inf")

            # Tokenize
            tokens = self._tokenizer(clean_text)
            if not tokens or len(tokens) == 0:
                return float("-inf")

            # Convert to tensor
            token_tensor = torch.tensor([tokens], dtype=torch.int32, device=self.device)
            emission_tensor = emissions.to(self.device)

            # Use torchaudio forced_align
            aligned_tokens, scores = torchaudio.functional.forced_align(
                emission_tensor, token_tensor, blank=0
            )

            # Compute average score
            if scores.numel() > 0:
                # scores is [1, T] — average over aligned frames
                avg_score = scores.mean().item()
                return avg_score
            else:
                return float("-inf")

        except Exception as e:
            logger.debug(f"Scoring failed for '{text}': {e}")
            return float("-inf")

    def score_candidates_batch(
        self,
        waveforms: List[torch.Tensor],
        candidate_lists: List[List[str]],
        languages: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float]]]:
        """Batch scoring for multiple audio-candidate pairs."""
        if languages is None:
            languages = ["en"] * len(waveforms)

        results = []
        for waveform, candidates, lang in zip(waveforms, candidate_lists, languages):
            scored = self.score_candidates(waveform, candidates, lang)
            results.append(scored)

        return results


class HeuristicScorer:
    """Fallback scorer using heuristic rules."""

    def score_candidates(
        self,
        candidates: List[str],
        language: str = "en",
    ) -> List[Tuple[str, float]]:
        """Score candidates using heuristic rules."""
        results = []
        for i, candidate in enumerate(candidates):
            score = 0.0

            # Prefer candidates without hyphens (more natural in speech)
            if "-" not in candidate:
                score += 1.0

            # Prefer shorter candidates slightly
            score -= len(candidate) * 0.001

            # Prefer candidates that are closer to the first (most common) option
            score -= i * 0.1

            results.append((candidate, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results
