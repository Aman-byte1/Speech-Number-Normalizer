"""
Unit tests for the speech number normalization package.

Tests cover:
  - Number detection (all categories)
  - Candidate generation (multiple languages)
  - Normalizer pipeline (text-only mode)
  - Edge cases
"""

import pytest

from speech_number_norm.detector import NumberDetector, NumberSpan, NumberCategory
from speech_number_norm.candidates import CandidateGenerator
from speech_number_norm.normalizer import SpeechNumberNormalizer


# ===================================================================
# Number Detector Tests
# ===================================================================

class TestNumberDetector:
    """Tests for NumberDetector."""

    def setup_method(self):
        self.detector = NumberDetector()

    # --- Cardinal ---
    def test_simple_cardinal(self):
        spans = self.detector.detect("There are 42 apples", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.CARDINAL
        assert spans[0].value == 42
        assert spans[0].raw_text == "42"

    def test_large_cardinal_with_commas(self):
        spans = self.detector.detect("Population is 1,000,000", "en")
        assert len(spans) == 1
        assert spans[0].value == 1000000

    def test_multiple_cardinals(self):
        spans = self.detector.detect("Buy 3 items for 5 dollars", "en")
        # Should detect at least the numbers
        assert len(spans) >= 2

    # --- Currency ---
    def test_dollar_currency(self):
        spans = self.detector.detect("The price is $31", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.CURRENCY
        assert spans[0].raw_text == "$31"

    def test_euro_currency(self):
        spans = self.detector.detect("It costs €500", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.CURRENCY

    def test_currency_with_decimals(self):
        spans = self.detector.detect("Total: $19.99", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.CURRENCY

    def test_currency_after_number(self):
        spans = self.detector.detect("It costs 500€", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.CURRENCY

    # --- Percentage ---
    def test_percentage(self):
        spans = self.detector.detect("A 50% increase", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.PERCENTAGE
        assert spans[0].value == 50

    def test_decimal_percentage(self):
        spans = self.detector.detect("Growth of 3.5%", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.PERCENTAGE
        assert spans[0].value == 3.5

    # --- Date ---
    def test_iso_date(self):
        spans = self.detector.detect("Born on 2024-01-15", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.DATE

    def test_slash_date(self):
        spans = self.detector.detect("Date: 01/15/2024", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.DATE

    # --- Time ---
    def test_time(self):
        spans = self.detector.detect("Meeting at 3:30 PM", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.TIME

    def test_24h_time(self):
        spans = self.detector.detect("Departure at 14:00", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.TIME

    # --- Ordinal ---
    def test_ordinal(self):
        spans = self.detector.detect("She is the 3rd child", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.ORDINAL
        assert spans[0].value == 3

    def test_ordinal_21st(self):
        spans = self.detector.detect("The 21st century", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.ORDINAL
        assert spans[0].value == 21

    # --- Year ---
    def test_year(self):
        spans = self.detector.detect("In the year 1984", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.YEAR
        assert spans[0].value == 1984

    def test_year_2024(self):
        spans = self.detector.detect("Welcome to 2024", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.YEAR

    # --- Decimal ---
    def test_decimal(self):
        spans = self.detector.detect("Pi is approximately 3.14", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.DECIMAL
        assert abs(spans[0].value - 3.14) < 0.001

    # --- Phone ---
    def test_phone(self):
        spans = self.detector.detect("Call 555-0123", "en")
        assert len(spans) == 1
        assert spans[0].category == NumberCategory.PHONE

    # --- Mixed ---
    def test_complex_sentence(self):
        text = "The $31 surcharge applies to the 3rd item, totaling $1,500.00"
        spans = self.detector.detect(text, "en")
        assert len(spans) >= 2  # At least the currencies

    def test_no_numbers(self):
        spans = self.detector.detect("Hello world, no numbers here", "en")
        assert len(spans) == 0


# ===================================================================
# Candidate Generator Tests
# ===================================================================

class TestCandidateGenerator:
    """Tests for CandidateGenerator."""

    def setup_method(self):
        self.gen = CandidateGenerator()

    def test_cardinal_english(self):
        span = NumberSpan(0, 2, "42", NumberCategory.CARDINAL, value=42)
        candidates = self.gen.generate(span, "en")
        assert len(candidates) >= 1
        assert any("forty" in c.lower() for c in candidates)

    def test_cardinal_large_number(self):
        span = NumberSpan(0, 5, "1,000", NumberCategory.CARDINAL, value=1000)
        candidates = self.gen.generate(span, "en")
        assert any("thousand" in c.lower() for c in candidates)

    def test_ordinal_english(self):
        span = NumberSpan(0, 3, "3rd", NumberCategory.ORDINAL, value=3)
        candidates = self.gen.generate(span, "en")
        assert any("third" in c.lower() for c in candidates)

    def test_currency_dollar(self):
        span = NumberSpan(0, 3, "$31", NumberCategory.CURRENCY, value=31,
                         parts={"symbol": "$", "amount": 31})
        candidates = self.gen.generate(span, "en")
        assert any("dollar" in c.lower() for c in candidates)
        assert any("thirty" in c.lower() for c in candidates)

    def test_percentage(self):
        span = NumberSpan(0, 3, "50%", NumberCategory.PERCENTAGE, value=50)
        candidates = self.gen.generate(span, "en")
        assert any("percent" in c.lower() for c in candidates)
        assert any("fifty" in c.lower() for c in candidates)

    def test_year_1984(self):
        span = NumberSpan(0, 4, "1984", NumberCategory.YEAR, value=1984)
        candidates = self.gen.generate(span, "en")
        assert any("nineteen" in c.lower() for c in candidates)
        # Should have both year-style and cardinal
        assert any("eighty" in c.lower() for c in candidates)

    def test_decimal(self):
        span = NumberSpan(0, 4, "3.14", NumberCategory.DECIMAL, value=3.14,
                         parts={"integer": "3", "fraction": "14"})
        candidates = self.gen.generate(span, "en")
        assert any("point" in c.lower() for c in candidates)
        assert any("three" in c.lower() for c in candidates)

    def test_time(self):
        span = NumberSpan(0, 4, "3:30", NumberCategory.TIME,
                         parts={"hour": 3, "minute": 30, "second": 0})
        candidates = self.gen.generate(span, "en")
        assert any("three" in c.lower() and "thirty" in c.lower() for c in candidates)

    # --- Multilingual ---
    def test_cardinal_spanish(self):
        span = NumberSpan(0, 2, "42", NumberCategory.CARDINAL, value=42)
        candidates = self.gen.generate(span, "es")
        assert len(candidates) >= 1
        assert any("cuarenta" in c.lower() for c in candidates)

    def test_cardinal_french(self):
        span = NumberSpan(0, 2, "42", NumberCategory.CARDINAL, value=42)
        candidates = self.gen.generate(span, "fr")
        assert len(candidates) >= 1
        assert any("quarante" in c.lower() for c in candidates)

    def test_cardinal_german(self):
        span = NumberSpan(0, 2, "42", NumberCategory.CARDINAL, value=42)
        candidates = self.gen.generate(span, "de")
        assert len(candidates) >= 1
        assert any("zweiundvierzig" in c.lower() or "zwei" in c.lower() for c in candidates)

    def test_always_returns_candidates(self):
        """Ensure we always get at least one candidate."""
        span = NumberSpan(0, 1, "0", NumberCategory.CARDINAL, value=0)
        candidates = self.gen.generate(span, "en")
        assert len(candidates) >= 1


# ===================================================================
# Normalizer Pipeline Tests (text-only mode)
# ===================================================================

class TestNormalizer:
    """Tests for SpeechNumberNormalizer in text-only mode."""

    def setup_method(self):
        self.normalizer = SpeechNumberNormalizer(use_audio=False)

    def test_basic_normalization(self):
        result = self.normalizer.normalize("I have 42 apples", language="en")
        assert "42" not in result.normalized_text
        assert "forty" in result.normalized_text.lower()

    def test_currency_normalization(self):
        result = self.normalizer.normalize("The price is $31", language="en")
        assert "$31" not in result.normalized_text
        assert "thirty" in result.normalized_text.lower()

    def test_no_numbers(self):
        result = self.normalizer.normalize("Hello world", language="en")
        assert result.normalized_text == "Hello world"
        assert len(result.replacements) == 0

    def test_multiple_numbers(self):
        result = self.normalizer.normalize("Buy 3 items for $50", language="en")
        assert "3" not in result.normalized_text
        assert "$50" not in result.normalized_text

    def test_preserves_surrounding_text(self):
        result = self.normalizer.normalize("Hello 42 world", language="en")
        assert result.normalized_text.startswith("Hello")
        assert result.normalized_text.endswith("world")

    def test_spanish_normalization(self):
        result = self.normalizer.normalize("Hay 42 manzanas", language="es")
        assert "42" not in result.normalized_text
        assert "cuarenta" in result.normalized_text.lower()

    def test_french_normalization(self):
        result = self.normalizer.normalize("Il y a 42 pommes", language="fr")
        assert "42" not in result.normalized_text
        assert "quarante" in result.normalized_text.lower()

    def test_result_metadata(self):
        result = self.normalizer.normalize("Price: $31", language="en")
        assert result.original_text == "Price: $31"
        assert result.language == "en"
        assert result.processing_time >= 0
        assert len(result.spans) >= 1
        assert len(result.replacements) >= 1


# ===================================================================
# Edge Cases
# ===================================================================

class TestEdgeCases:
    """Edge case tests."""

    def setup_method(self):
        self.detector = NumberDetector()
        self.normalizer = SpeechNumberNormalizer(use_audio=False)

    def test_empty_text(self):
        spans = self.detector.detect("", "en")
        assert len(spans) == 0

    def test_single_digit(self):
        result = self.normalizer.normalize("I have 1 apple", language="en")
        assert "1" not in result.normalized_text

    def test_zero(self):
        result = self.normalizer.normalize("Temperature is 0 degrees", language="en")
        assert "0" not in result.normalized_text
        assert "zero" in result.normalized_text.lower()

    def test_very_large_number(self):
        result = self.normalizer.normalize("Population: 1,000,000", language="en")
        assert "1,000,000" not in result.normalized_text
        assert "million" in result.normalized_text.lower()

    def test_negative_not_detected(self):
        """Negative numbers are not in current scope."""
        spans = self.detector.detect("It was -5 degrees", "en")
        # Should detect the 5 as cardinal (- is not part of our patterns)
        cardinal_spans = [s for s in spans if s.category == NumberCategory.CARDINAL]
        assert len(cardinal_spans) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
