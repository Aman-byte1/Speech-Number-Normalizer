import logging
import os
import tempfile
import gradio as gr

from speech_number_norm.normalizer import SpeechNumberNormalizer
from speech_number_norm.lang_config import supported_languages, LANGUAGE_CONFIGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language display names for the dropdown
LANGUAGE_CHOICES = {
    code: f"{config.name} ({code})"
    for code, config in LANGUAGE_CONFIGS.items()
}

# Example inputs for different languages
EXAMPLES = [
    ["The total cost is $31 for 3 items, ordered on 01/15/2024.", None, "English (en)"],
    ["She was born on 1984-06-15 and is the 3rd of 5 children.", None, "English (en)"],
    ["The temperature reached 98.6°F, which is 37% above normal.", None, "English (en)"],
    ["Call us at 555-0123 for a 50% discount on orders over $1,000.", None, "English (en)"],
    ["El precio es €500 por 2 unidades, con un 15% de descuento.", None, "Spanish (es)"],
    ["Le rendez-vous est le 15/03/2024 à 14:30.", None, "French (fr)"],
    ["Der Preis beträgt €1.500 für 3 Artikel.", None, "German (de)"],
    ["Il costo è €200 per 4 articoli, con il 10% di sconto.", None, "Italian (it)"],
]


def create_normalizer():
    """Create the normalizer instance (text-only mode for demo speed)."""
    return SpeechNumberNormalizer(device="auto", use_audio=True)


# Global normalizer instance
_normalizer = None


def get_normalizer():
    """Get or create the global normalizer."""
    global _normalizer
    if _normalizer is None:
        _normalizer = create_normalizer()
    return _normalizer


def normalize_text(
    text: str,
    audio,
    language: str,
    use_audio_scoring: bool,
) -> tuple:
    """
    Main demo function: normalize numbers in text.
    
    Args:
        text: Input text with numbers.
        audio: Audio file (from upload or microphone).
        language: Language code.
        use_audio_scoring: Whether to use audio for scoring.
        
    Returns:
        Tuple of (normalized_text, details_markdown).
    """
    if not text.strip():
        return "Please enter some text.", ""

    # Extract language code from display name if needed
    lang_code = language
    for code, display in LANGUAGE_CHOICES.items():
        if display == language or code == language:
            lang_code = code
            break

    normalizer = get_normalizer()

    # Handle audio path
    audio_path = None
    if audio is not None and use_audio_scoring:
        if isinstance(audio, str):
            audio_path = audio
        elif isinstance(audio, tuple):
            # Gradio returns (sample_rate, numpy_array) for microphone
            sr, data = audio
            import numpy as np
            import torch
            import torchaudio

            audio_path = os.path.join(tempfile.gettempdir(), "demo_audio.wav")
            waveform = torch.tensor(data, dtype=torch.float32)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.dim() == 2:
                waveform = waveform.T
            torchaudio.save(audio_path, waveform, sr)

    if not use_audio_scoring:
        normalizer_to_use = SpeechNumberNormalizer(use_audio=False)
        result = normalizer_to_use.normalize(text, None, lang_code)
    else:
        result = normalizer.normalize(text, audio_path, lang_code)

    details = _build_details_md(result)

    return result.normalized_text, details


def _build_details_md(result) -> str:
    lines = []
    lines.append(f"### Normalization Details")
    lines.append(f"")
    lines.append(f"**Language:** {result.language}")
    lines.append(f"**Processing time:** {result.processing_time:.3f}s")
    lines.append(f"**Number spans detected:** {len(result.spans)}")
    lines.append("")

    if not result.replacements:
        lines.append("*No numbers detected in the input text.*")
        return "\n".join(lines)

    for i, rep in enumerate(result.replacements):
        span = rep["span"]
        lines.append(f"---")
        lines.append(f"#### Span {i+1}: `{span.raw_text}`")
        lines.append(f"- **Category:** {span.category.name}")
        lines.append(f"- **Value:** {span.value}")
        lines.append(f"- **Chosen:** `{rep['chosen']}`")
        lines.append(f"")

        # Show top candidates with scores
        lines.append(f"| Rank | Candidate | Score |")
        lines.append(f"|------|-----------|-------|")
        for j, (cand, score) in enumerate(rep["scores"][:6]):
            marker = " ✅" if cand == rep["chosen"] else ""
            lines.append(f"| {j+1} | {cand}{marker} | {score:+.4f} |")
        lines.append("")

    return "\n".join(lines)


def build_demo():
    """Build and return the Gradio demo interface."""

    with gr.Blocks(
        title="Speech Number Normalizer",
    ) as demo:
        gr.HTML("""
            <div class="main-title">🔢 Speech Number Normalizer</div>
            <div class="subtitle">
                Convert written numbers to spoken form using audio context
                <br/>Supports 20+ languages · Powered by MMS + num2words
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="📝 Input Text",
                    placeholder="Enter text with numbers, e.g., 'The price is $31 for 3 items'",
                    lines=3,
                )

                language_input = gr.Dropdown(
                    label="🌍 Language",
                    choices=list(LANGUAGE_CHOICES.values()),
                    value=LANGUAGE_CHOICES.get("en", "English (en)"),
                )

                audio_input = gr.Audio(
                    label="🎙️ Audio (optional)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )

                use_audio = gr.Checkbox(
                    label="🔊 Use audio for scoring",
                    value=True,
                    info="When disabled, uses heuristic scoring (faster, no audio needed)",
                )

                normalize_btn = gr.Button(
                    "🚀 Normalize",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="✅ Normalized Text",
                    lines=3,
                    interactive=False,
                    elem_classes=["output-box"],
                )

                details_output = gr.Markdown(
                    label="📊 Details",
                )

        # Examples
        gr.Examples(
            examples=EXAMPLES,
            inputs=[text_input, audio_input, language_input],
            label="📌 Example Inputs (click to load)",
        )

        # Event handlers
        normalize_btn.click(
            fn=normalize_text,
            inputs=[text_input, audio_input, language_input, use_audio],
            outputs=[output_text, details_output],
        )

        # Also trigger on Enter key in text input
        text_input.submit(
            fn=normalize_text,
            inputs=[text_input, audio_input, language_input, use_audio],
            outputs=[output_text, details_output],
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
    )
    css = """
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: 800;
        margin-bottom: 0.2em;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 1.1em;
        margin-bottom: 1.5em;
    }
    .output-box {
        font-size: 1.15em;
        line-height: 1.6;
    }
    """
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=theme,
        css=css,
    )
