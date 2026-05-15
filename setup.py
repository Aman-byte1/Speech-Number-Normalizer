from setuptools import setup, find_packages

setup(
    name="speech-number-norm",
    version="0.1.0",
    description="Multilingual speech number normalization: convert written numbers to spoken form using audio context",
    author="Geama",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "num2words>=0.5.13",
        "gradio>=4.0.0",
        "regex>=2023.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "speech-number-norm=speech_number_norm.normalizer:main",
        ],
    },
)
