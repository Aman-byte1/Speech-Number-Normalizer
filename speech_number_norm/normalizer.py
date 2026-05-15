import argparse
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

from speech_number_norm.candidates import CandidateGenerator
from speech_number_norm.detector import NumberDetector, NumberSpan
from speech_number_norm.lang_config import get_config, supported_languages
from speech_number_norm.scorer import CTCScorer, HeuristicScorer
from speech_number_norm.utils import clean_text, load_audio, estimate_number_audio_region

logger = logging.getLogger(__name__)


@dataclass
class NormalizationResult:
    original_text: str
    normalized_text: str
    language: str
    spans: List[NumberSpan]
    replacements: List[dict]
    processing_time: float


class SpeechNumberNormalizer:
    """Main entry point for speech number normalization."""

    def __init__(self, device: str = "auto", use_audio: bool = True):
        self.detector = NumberDetector()
        self.candidate_gen = CandidateGenerator()
        self.use_audio = use_audio

        if use_audio:
            self.scorer = CTCScorer(device=device)
        else:
            self.scorer = None

        self.heuristic_scorer = HeuristicScorer()

    def normalize(
        self,
        text: str,
        audio_path: Optional[str] = None,
        language: str = "en",
    ) -> NormalizationResult:
        """Normalize numbers in text to their spoken form."""
        start_time = time.time()
        text = clean_text(text)

        # Step 1: Detect number spans
        spans = self.detector.detect(text, language)

        if not spans:
            return NormalizationResult(
                original_text=text,
                normalized_text=text,
                language=language,
                spans=[],
                replacements=[],
                processing_time=time.time() - start_time,
            )

        # Step 2: Load audio if available
        waveform = None
        audio_duration = 0.0
        if audio_path and self.use_audio and os.path.exists(audio_path):
            try:
                waveform, sr = load_audio(audio_path)
                audio_duration = waveform.shape[1] / sr
            except Exception as e:
                logger.warning(f"Failed to load audio '{audio_path}': {e}")
                waveform = None

        # Step 3: For each span, generate candidates and score
        replacements = []
        for span in spans:
            candidates = self.candidate_gen.generate(span, language)

            if waveform is not None and self.scorer is not None and len(candidates) > 1:
                # Extract the relevant audio region
                start_t, end_t = estimate_number_audio_region(
                    text, span.start, span.end, audio_duration
                )
                sr = 16000
                start_sample = int(start_t * sr)
                end_sample = int(end_t * sr)
                region = waveform[:, start_sample:end_sample]

                # Score candidates against audio
                if region.shape[1] > 0:
                    scored = self.scorer.score_candidates(region, candidates, language)
                else:
                    scored = self.heuristic_scorer.score_candidates(candidates, language)
            else:
                # Text-only mode: use heuristic scoring
                scored = self.heuristic_scorer.score_candidates(candidates, language)

            best_candidate = scored[0][0] if scored else candidates[0]

            replacements.append({
                "span": span,
                "candidates": candidates,
                "scores": scored,
                "chosen": best_candidate,
            })

        # Step 4: Replace spans in text (process from end to preserve indices)
        normalized = text
        for rep in reversed(replacements):
            span = rep["span"]
            chosen = rep["chosen"]
            normalized = normalized[:span.start] + chosen + normalized[span.end:]

        elapsed = time.time() - start_time

        return NormalizationResult(
            original_text=text,
            normalized_text=normalized,
            language=language,
            spans=spans,
            replacements=replacements,
            processing_time=elapsed,
        )

    def normalize_batch(
        self,
        texts: List[str],
        audio_paths: Optional[List[str]] = None,
        language: str = "en",
    ) -> List[NormalizationResult]:
        """Normalize a batch of (text, audio) pairs."""
        if audio_paths is None:
            audio_paths = [None] * len(texts)

        results = []
        for text, audio in zip(texts, audio_paths):
            result = self.normalize(text, audio, language)
            results.append(result)

        return results


def main():
    """CLI entry point for speech number normalization."""
    parser = argparse.ArgumentParser(
        description="Speech Number Normalization: Convert written numbers to spoken form"
    )
    parser.add_argument("text", help="Input text with numbers to normalize")
    parser.add_argument(
        "--audio", "-a", default=None,
        help="Path to corresponding audio file (optional)"
    )
    parser.add_argument(
        "--language", "-l", default="en",
        help=f"Language code (supported: {', '.join(supported_languages())})"
    )
    parser.add_argument(
        "--device", "-d", default="auto",
        help="Device for audio model: auto, cuda, cpu"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Disable audio scoring (text-only mode)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    normalizer = SpeechNumberNormalizer(
        device=args.device,
        use_audio=not args.no_audio,
    )

    result = normalizer.normalize(
        text=args.text,
        audio_path=args.audio,
        language=args.language,
    )

    print(f"\n{'='*60}")
    print(f"  Original:   {result.original_text}")
    print(f"  Normalized: {result.normalized_text}")
    print(f"  Language:   {result.language}")
    print(f"  Time:       {result.processing_time:.3f}s")
    print(f"{'='*60}")

    if result.replacements:
        print("\n  Replacements:")
        for rep in result.replacements:
            span = rep["span"]
            print(f"    '{span.raw_text}' ({span.category.name}) -> '{rep['chosen']}'")
            if args.verbose:
                print(f"      Candidates & scores:")
                for cand, score in rep["scores"][:5]:
                    print(f"        {score:+.4f}  {cand}")
    print()


if __name__ == "__main__":
    main()
