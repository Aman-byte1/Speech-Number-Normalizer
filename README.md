# 🔢 Speech Number Normalizer

**GitHub Repository:** [https://github.com/Aman-byte1/Speech-Number-Normalizer](https://github.com/Aman-byte1/Speech-Number-Normalizer)

**Multilingual speech number normalization**: Given a (speech, text) pair, convert numbers in text to the form spoken in the audio. Designed for preprocessing massive multilingual TTS/voice-conversion training datasets.

## Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
  - [Python API](#python-api)
  - [Command Line](#command-line)
  - [Gradio Demo](#gradio-demo)
  - [Evaluation](#evaluation)
- [Algorithm Design](#algorithm-design)
  - [Architecture Overview](#architecture-overview)
  - [Stage 1: Number Detection](#stage-1-number-detection)
  - [Stage 2: Candidate Generation](#stage-2-candidate-generation)
  - [Stage 3: Audio Scoring](#stage-3-audio-scoring)
- [Design Decisions & Tradeoffs](#design-decisions--tradeoffs)
- [Supported Languages](#supported-languages)
- [Scope & Limitations](#scope--limitations)
- [Performance & Scaling](#performance--scaling)

---

## Quick Start

```bash
# Install
pip install -r requirements.txt
pip install -e .

# Text-only normalization (no audio needed)
python -m speech_number_norm.normalizer "The price is \$31 for 3 items" --no-audio -l en

# With audio context
python -m speech_number_norm.normalizer "The price is \$31" --audio utterance.wav -l en

# Launch interactive demo
python demo.py
```

## Installation

### Requirements
- Python ≥ 3.9
- PyTorch ≥ 2.0 (with CUDA recommended for audio scoring)

```bash
# Clone the repository
git clone https://github.com/Aman-byte1/Speech-Number-Normalizer.git
cd word_totext

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### GPU Support (recommended)
For audio-based scoring, a CUDA-capable GPU is recommended but not required.
The system automatically falls back to CPU or text-only mode when GPU is unavailable.

---

## Usage

### Python API

```python
from speech_number_norm import SpeechNumberNormalizer

# Initialize (text-only mode — fast, no GPU needed)
normalizer = SpeechNumberNormalizer(use_audio=False)

result = normalizer.normalize(
    text="The surcharge will be $31",
    language="en"
)
print(result.normalized_text)
# "The surcharge will be thirty one dollars"

# With audio context (requires PyTorch + torchaudio)
normalizer = SpeechNumberNormalizer(device="cuda", use_audio=True)

result = normalizer.normalize(
    text="The surcharge will be $31",
    audio_path="utterance.wav",
    language="en"
)
print(result.normalized_text)
# "The surcharge will be thirty-one dollars"  (matches audio)

# Inspect details
for rep in result.replacements:
    print(f"  {rep['span'].raw_text} -> {rep['chosen']}")
    for candidate, score in rep['scores'][:3]:
        print(f"    {score:+.4f}  {candidate}")
```

### Batch Processing

```python
# Process multiple utterances
results = normalizer.normalize_batch(
    texts=["Price: $31", "Date: 01/15/2024", "Score: 98.6%"],
    audio_paths=["audio1.wav", "audio2.wav", "audio3.wav"],
    language="en"
)
for r in results:
    print(r.normalized_text)
```

### Command Line

```bash
# Basic usage (text-only)
python -m speech_number_norm.normalizer "Buy 3 items for \$50" --no-audio

# With audio
python -m speech_number_norm.normalizer "The price is \$31" -a audio.wav -l en

# Verbose mode (show all candidates and scores)
python -m speech_number_norm.normalizer "She was born in 1984" -v --no-audio

# Different language
python -m speech_number_norm.normalizer "El precio es €500" -l es --no-audio
```

### Gradio Demo

```bash
python demo.py
# Opens at http://localhost:7860
```

### Evaluation

An evaluation script `evaluate.py` is included to benchmark the system using the LibriSpeech dataset. It performs a "closed-loop" evaluation by:
1. Loading a sample of normalized speech (words).
2. Denormalizing it back to digits (e.g., "twenty two" -> "22").
3. Running the normalizer to see if it can recover the original spoken form using the audio.

```bash
# Run evaluation (requires datasets and word2number libraries)
pip install datasets word2number
python evaluate.py
```

The demo provides:
- Text input with number detection
- Audio upload or microphone recording
- Language selection (20+ languages)
- Detailed results showing all candidates and scores

---

## Algorithm Design

### Architecture Overview

The system follows a **generate-and-score** paradigm:

```
Input Text ──→ [Detect Numbers] ──→ [Generate Candidates] ──→ [Score vs Audio] ──→ Normalized Text
                                                                      ↑
Input Audio ──────────────────────────────────────────────────────────┘
```

**Three-stage pipeline:**

1. **Detection** — Regex-based identification and classification of numeric expressions
2. **Generation** — Produce multiple plausible spoken-form candidates using `num2words`
3. **Scoring** — Rank candidates by CTC forced alignment log-probability against the audio

This approach was chosen over alternatives for specific reasons:

| Approach | Speed | Accuracy | Multilingual | Why not? |
|----------|-------|----------|--------------|----------|
| Full ASR → diff | Slow | High | Limited | Requires high-quality ASR per language |
| Multimodal LLM | Very slow | Very high | Good | Way too slow/expensive for 100M+ samples |
| **Generate + CTC Score** | **Fast** | **Good** | **Excellent** | **← Chosen** |
| WFST only (no audio) | Fastest | Low | Medium | Can't disambiguate without audio |

### Stage 1: Number Detection

**File:** `speech_number_norm/detector.py`

Uses priority-ordered regex patterns to detect and classify 10 categories of numeric expressions:

| Category | Examples | Regex Strategy |
|----------|----------|----------------|
| Currency | `$31`, `€500`, `£50.99`, `500€` | Symbol + number patterns |
| Percentage | `50%`, `3.5%` | Number + `%` |
| Date (ISO) | `2024-01-15` | `YYYY-MM-DD` |
| Date (slash) | `01/15/2024`, `15/01/2024` | `DD/MM/YYYY` variants |
| Date (named) | `Jan 15, 2024` | Month name + numbers |
| Time | `3:30 PM`, `14:00` | `HH:MM` with optional AM/PM |
| Ordinal | `1st`, `3rd`, `21st` | Number + ordinal suffix |
| Phone | `555-0123` | Digit groups with separators |
| Decimal | `3.14`, `0.5` | Number + `.` + number |
| Year | `1984`, `2024` | Standalone 4-digit (1000-2999) |
| Cardinal | `42`, `1,000,000` | Plain integers |

**Overlap resolution:** More specific patterns (currency, date) take priority over generic ones (cardinal). The detector resolves conflicts by keeping the higher-priority, longer match.

### Stage 2: Candidate Generation

**File:** `speech_number_norm/candidates.py`

For each detected span, generates multiple spoken-form candidates using:

1. **`num2words` library** — Primary generator supporting 50+ languages with cardinal, ordinal, currency, and year modes
2. **Custom rules** — Hyphenation variants, currency word insertion, date formatting, digit-by-digit alternatives
3. **LRU caching** — 65,536-entry cache for `num2words` calls (identical numbers appear frequently in large datasets)

**Example candidates for `$31` in English:**
```
thirty-one dollars          (num2words currency mode)
thirty one dollars          (no-hyphen variant)
thirty-one                  (without currency word)
thirty one                  (simplest form)
```

**Example candidates for `1984` as year:**
```
nineteen eighty-four        (year-style: split at century)
nineteen eighty four        (no-hyphen variant)
one thousand nine hundred eighty-four  (full cardinal)
```

### Stage 3: Audio Scoring

**File:** `speech_number_norm/scorer.py`

Uses **Meta's MMS (Massively Multilingual Speech)** CTC forced alignment model via `torchaudio`:

1. **Audio region estimation** — Uses proportional character-to-time mapping to isolate the relevant audio segment, reducing computation
2. **CTC emission extraction** — Single forward pass through the MMS model
3. **Forced alignment scoring** — For each candidate, compute CTC log-probability using `torchaudio.functional.forced_align()`
4. **Selection** — Candidate with highest average log-probability wins

**Why MMS?**
- Supports **1,100+ languages** — by far the widest multilingual coverage
- CTC architecture enables **efficient scoring** (no autoregressive decoding)
- Available as a single pre-trained model via `torchaudio.pipelines.MMS_FA`
- Uses `uroman` for universal script normalization

**Fallback:** When audio is unavailable or scoring fails, a heuristic scorer ranks candidates by:
- Preference for no-hyphen forms (more natural in speech)
- Shorter candidates (simpler verbalizations)
- First-generated candidate priority (most common pattern)

---

## Design Decisions & Tradeoffs

### What Worked Well

1. **`num2words` + custom rules** — Excellent coverage for candidate generation. The library supports 50+ languages out of the box, and custom post-processing handles variants like hyphenation and currency words effectively.

2. **Regex detection with priority ordering** — Fast, deterministic, and easy to extend. Currency/date patterns correctly override cardinal patterns (e.g., `$31` is detected as currency, not cardinal).

3. **LRU caching** — Significant speedup for large datasets where the same numbers appear repeatedly.

4. **Text-only mode** — Essential for rapid bulk processing. Even without audio, the heuristic ranking produces reasonable results for most cases.

### What Was Challenging

1. **Date format ambiguity** — `01/02/2024` could be January 2 (MDY) or February 1 (DMY). Resolution depends on locale, handled via `lang_config.py`.

2. **Year vs cardinal disambiguation** — `1984` could be a year or a cardinal. Currently uses position heuristics (standalone 4-digit = year), which works ~90% of the time but can fail in edge cases.

3. **Non-Latin scripts** — MMS expects romanized text for forced alignment. For languages like Arabic, Hindi, Chinese, the system relies on `uroman` or basic transliteration, which may lose phonetic precision.

4. **Currency symbol ambiguity** — `$` could be USD, CAD, AUD, etc. The system defaults to the most common currency for the detected language.

### What's In Scope vs Out of Scope

| In Scope | Out of Scope |
|----------|-------------|
| Cardinals, ordinals | Roman numerals (IV, XIV) |
| Currencies ($, €, £, ¥, etc.) | Mathematical expressions |
| Dates (ISO, slash, named) | Fractions as text (1/3) |
| Times (12h, 24h) | Measurement units (5'6") |
| Percentages | Alphanumeric codes (B-52) |
| Decimals | Negative numbers |
| Year expressions | Phone number formatting by country |
| Phone/digit sequences | |

### Languages: What Works Well vs Badly

| Tier | Languages | Quality | Notes |
|------|-----------|---------|-------|
| **Excellent** | English, Spanish, French, German, Italian, Portuguese | Full coverage | All num2words features + custom rules |
| **Good** | Russian, Dutch, Swedish, Polish, Turkish, Ukrainian | Cardinal + ordinal | Currency support varies |
| **Fair** | Chinese, Japanese, Korean, Arabic, Hindi | Cardinals work | Date/currency patterns need tuning |
| **Basic** | Amharic, Vietnamese, Thai, Indonesian, Hebrew | Cardinal only | Limited num2words features |

---

## Supported Languages

22 languages with explicit configuration, 50+ via `num2words` fallback:

`am` (Amharic), `ar` (Arabic), `de` (German), `en` (English), `es` (Spanish), `fr` (French), `he` (Hebrew), `hi` (Hindi), `id` (Indonesian), `it` (Italian), `ja` (Japanese), `ko` (Korean), `nl` (Dutch), `pl` (Polish), `pt` (Portuguese), `ru` (Russian), `sv` (Swedish), `th` (Thai), `tr` (Turkish), `uk` (Ukrainian), `vi` (Vietnamese), `zh` (Chinese)

---

## Scope & Limitations

### Known Limitations

1. **Audio region estimation is approximate** — Uses character-position-to-time proportional mapping. Works well for simple sentences but may be inaccurate for long texts with uneven speech rates.

2. **No context-aware disambiguation** — The system doesn't use surrounding text context to resolve ambiguities (e.g., "1984" as a year in "published in 1984" vs a number in "scored 1984 points").

3. **Currency/date ambiguity** — Some symbols and formats are inherently ambiguous without broader context.

4. **Non-Latin alignment quality** — CTC scoring quality depends on `uroman` transliteration, which may not perfectly preserve phonetic information for all scripts.

### Potential Improvements

- **ASR pre-pass**: Run lightweight ASR first to get rough transcript, then use it to narrow the audio region for each number span
- **Context window**: Use surrounding words to disambiguate year vs cardinal
- **WFST integration**: Use NeMo's WFST grammars for more linguistically precise candidate generation
- **Language-specific date/currency patterns**: Add regex patterns tuned per language
- **Parallel batch processing**: Use `torch.multiprocessing` for data-parallel GPU scoring

---

## Performance & Scaling

### Speed Characteristics

| Mode | Speed (per utterance) | GPU Required | Accuracy |
|------|----------------------|--------------|----------|
| Text-only | ~1ms | No | Good (heuristic) |
| Audio scoring | ~50-100ms | Recommended | Best |

### Scaling Strategies for 100M+ Samples

1. **Text-only preprocessing** — For unambiguous numbers (single candidate), skip audio scoring entirely
2. **LRU caching** — Avoid recomputing identical `num2words` calls
3. **Batch GPU inference** — Process multiple utterances in parallel
4. **CPU/GPU pipelining** — Text detection on CPU while audio scores on GPU
5. **Sharded processing** — Split dataset across multiple workers/machines

### Example: Processing 1M utterances

```python
from speech_number_norm import SpeechNumberNormalizer
from concurrent.futures import ProcessPoolExecutor

normalizer = SpeechNumberNormalizer(device="cuda", use_audio=True)

# Process in batches
for batch in dataset.iter(batch_size=256):
    results = normalizer.normalize_batch(
        texts=batch["text"],
        audio_paths=batch["audio"],
        language="en"
    )
```

---

## Project Structure

```
word_totext/
├── speech_number_norm/       # Main Python package
│   ├── __init__.py           # Public API exports
│   ├── detector.py           # Regex-based number span detection
│   ├── candidates.py         # Candidate verbalization generator
│   ├── scorer.py             # CTC audio scorer (MMS)
│   ├── normalizer.py         # Main pipeline orchestrator
│   ├── lang_config.py        # Per-language configuration
│   └── utils.py              # Audio/text utilities
├── demo.py                   # Gradio interactive demo
├── tests/
│   └── test_normalizer.py    # Unit tests
├── setup.py                  # Package setup
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

---

## License

MIT
