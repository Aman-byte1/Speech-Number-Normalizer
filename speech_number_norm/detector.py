"""
Number span detection in text.

Uses regex patterns to find and classify numeric expressions:
currencies, percentages, dates, times, ordinals, decimals, cardinals,
and digit sequences (phone numbers, codes).
"""

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple

from speech_number_norm.lang_config import get_config, LanguageConfig


class NumberCategory(Enum):
    """Classification of detected number spans."""
    CARDINAL = auto()       # 42, 1000, 1,000,000
    ORDINAL = auto()        # 1st, 2nd, 3rd, 21st
    CURRENCY = auto()       # $31, €500, £50.99
    PERCENTAGE = auto()     # 50%, 3.5%
    DECIMAL = auto()        # 3.14, 0.5
    DATE = auto()           # 2024-01-15, 01/15/2024
    TIME = auto()           # 3:30, 14:00
    YEAR = auto()           # 1984, 2024 (standalone 4-digit)
    PHONE = auto()          # 555-0123, digit sequences with dashes
    DIGIT_SEQUENCE = auto() # Other sequences best read digit-by-digit


@dataclass
class NumberSpan:
    """A detected numeric expression in text."""
    start: int              # Start character index in original text
    end: int                # End character index in original text
    raw_text: str           # Original text of the span (e.g. "$31")
    category: NumberCategory
    value: Optional[float] = None          # Parsed numeric value (if applicable)
    parts: Optional[dict] = None           # Structured parts (for dates, times, currencies)

    def __repr__(self):
        return f"NumberSpan({self.raw_text!r}, {self.category.name}, val={self.value})"


class NumberDetector:
    """
    Detects numeric expressions in text and classifies them.
    
    Uses a priority-ordered set of regex patterns. More specific patterns
    (currency, date) are checked before generic ones (cardinal) to avoid
    partial matches.
    """

    # ---------------------------------------------------------------
    # Currency symbols (escaped for regex)
    # ---------------------------------------------------------------
    _CURRENCY_SYMBOLS = r"[\$€£¥₹₽₩₪฿]"
    _CURRENCY_CODES = r"(?:USD|EUR|GBP|JPY|INR|RUB|KRW|ILS|THB|ZAR|SEK|CHF|ETB|CAD|AUD|NZD|BRL|CNY|MXN|SGD)"

    def __init__(self):
        """Initialize patterns. Patterns are compiled once and reused."""
        self._patterns = self._build_patterns()

    def detect(self, text: str, language: str = "en") -> List[NumberSpan]:
        """
        Detect all numeric expressions in text.
        
        Args:
            text: Input text to scan.
            language: ISO 639-1 language code.
            
        Returns:
            List of NumberSpan objects sorted by position, non-overlapping.
        """
        config = get_config(language)
        spans: List[NumberSpan] = []

        for category, pattern in self._patterns:
            for match in pattern.finditer(text):
                span = self._parse_match(match, category, config)
                if span is not None:
                    spans.append(span)

        # Remove overlapping spans (keep higher-priority / longer ones)
        spans = self._resolve_overlaps(spans)
        # Sort by position
        spans.sort(key=lambda s: s.start)
        return spans

    def _build_patterns(self) -> List[Tuple[NumberCategory, re.Pattern]]:
        """
        Build regex patterns in priority order.
        Higher priority patterns are listed first and win on overlap.
        """
        patterns = []

        # ----- CURRENCY (highest priority) -----
        # Symbol before: $31, $1,000.50, €500
        patterns.append((
            NumberCategory.CURRENCY,
            re.compile(
                rf"({self._CURRENCY_SYMBOLS})\s*(\d{{1,3}}(?:[,.\s]\d{{3}})*(?:[.,]\d{{1,2}})?)(?!\d)",
                re.UNICODE
            )
        ))
        # Symbol after: 500€, 1000₽
        patterns.append((
            NumberCategory.CURRENCY,
            re.compile(
                rf"(\d{{1,3}}(?:[,.\s]\d{{3}})*(?:[.,]\d{{1,2}})?)\s*({self._CURRENCY_SYMBOLS})",
                re.UNICODE
            )
        ))
        # Currency code: USD 500, 500 EUR
        patterns.append((
            NumberCategory.CURRENCY,
            re.compile(
                rf"({self._CURRENCY_CODES})\s*(\d{{1,3}}(?:[,.\s]\d{{3}})*(?:[.,]\d{{1,2}})?)"
                rf"|(\d{{1,3}}(?:[,.\s]\d{{3}})*(?:[.,]\d{{1,2}})?)\s*({self._CURRENCY_CODES})",
                re.UNICODE
            )
        ))

        # ----- PERCENTAGE -----
        patterns.append((
            NumberCategory.PERCENTAGE,
            re.compile(
                r"(\d+(?:[.,]\d+)?)\s*%",
                re.UNICODE
            )
        ))

        # ----- DATE (ISO format: 2024-01-15) -----
        patterns.append((
            NumberCategory.DATE,
            re.compile(
                r"(\d{4})-(\d{1,2})-(\d{1,2})",
                re.UNICODE
            )
        ))
        # Date with slashes: 01/15/2024 or 15/01/2024
        patterns.append((
            NumberCategory.DATE,
            re.compile(
                r"(\d{1,2})/(\d{1,2})/(\d{2,4})",
                re.UNICODE
            )
        ))
        # Date with month name: Jan 15, 2024 or 15 January 2024
        patterns.append((
            NumberCategory.DATE,
            re.compile(
                r"(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
                r"Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|"
                r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s*\d{2,4})"
                r"|"
                r"(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
                r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
                r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?),?\s*\d{2,4})",
                re.UNICODE | re.IGNORECASE
            )
        ))

        # ----- TIME -----
        patterns.append((
            NumberCategory.TIME,
            re.compile(
                r"(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(?:AM|PM|am|pm|a\.m\.|p\.m\.)?",
                re.UNICODE
            )
        ))

        # ----- ORDINAL -----
        patterns.append((
            NumberCategory.ORDINAL,
            re.compile(
                r"(\d+)(?:st|nd|rd|th|º|ª|ème|er|ère)\b",
                re.UNICODE | re.IGNORECASE
            )
        ))

        # ----- PHONE / DIGIT SEQUENCE (with dashes or spaces between digit groups) -----
        patterns.append((
            NumberCategory.PHONE,
            re.compile(
                r"\b(\d{2,4}[-.\s]\d{2,4}(?:[-.\s]\d{2,4})*)\b",
                re.UNICODE
            )
        ))

        # ----- DECIMAL -----
        patterns.append((
            NumberCategory.DECIMAL,
            re.compile(
                r"\b(\d+)\.(\d+)\b",
                re.UNICODE
            )
        ))

        # ----- YEAR (standalone 4-digit number that looks like a year) -----
        patterns.append((
            NumberCategory.YEAR,
            re.compile(
                r"(?<!\d)\b(1[0-9]{3}|20[0-9]{2}|2[1-9][0-9]{2})\b(?!\d)",
                re.UNICODE
            )
        ))

        # ----- CARDINAL (plain integers with optional thousands separators) -----
        patterns.append((
            NumberCategory.CARDINAL,
            re.compile(
                r"(?<!\d)(\d{1,3}(?:[,.\s]\d{3})*)\b(?!\.\d)",
                re.UNICODE
            )
        ))

        return patterns

    def _parse_match(
        self, match: re.Match, category: NumberCategory, config: LanguageConfig
    ) -> Optional[NumberSpan]:
        """Parse a regex match into a NumberSpan with structured data."""
        raw = match.group(0)
        start, end = match.start(), match.end()

        try:
            if category == NumberCategory.CURRENCY:
                parts = self._parse_currency(match, config)
                value = parts.get("amount", 0)
                return NumberSpan(start, end, raw, category, value=value, parts=parts)

            elif category == NumberCategory.PERCENTAGE:
                num_str = match.group(1).replace(",", ".")
                value = float(num_str)
                return NumberSpan(start, end, raw, category, value=value)

            elif category == NumberCategory.DATE:
                parts = self._parse_date(match, raw, config)
                return NumberSpan(start, end, raw, category, parts=parts)

            elif category == NumberCategory.TIME:
                hour = int(match.group(1))
                minute = int(match.group(2))
                second = int(match.group(3)) if match.group(3) else 0
                parts = {"hour": hour, "minute": minute, "second": second}
                return NumberSpan(start, end, raw, category, parts=parts)

            elif category == NumberCategory.ORDINAL:
                value = float(match.group(1))
                return NumberSpan(start, end, raw, category, value=value)

            elif category == NumberCategory.PHONE:
                digits = re.sub(r"[^\d]", "", raw)
                return NumberSpan(start, end, raw, category,
                                 parts={"digits": digits})

            elif category == NumberCategory.DECIMAL:
                value = float(raw)
                parts = {"integer": match.group(1), "fraction": match.group(2)}
                return NumberSpan(start, end, raw, category, value=value, parts=parts)

            elif category == NumberCategory.YEAR:
                value = float(match.group(1))
                return NumberSpan(start, end, raw, category, value=value)

            elif category == NumberCategory.CARDINAL:
                # Strip thousands separators
                clean = re.sub(r"[,.\s]", "", match.group(1))
                if not clean:
                    return None
                value = float(clean)
                return NumberSpan(start, end, raw, category, value=value)

            else:
                return NumberSpan(start, end, raw, category)

        except (ValueError, IndexError):
            return None

    def _parse_currency(self, match: re.Match, config: LanguageConfig) -> dict:
        """Extract currency symbol and amount from a currency match."""
        groups = match.groups()
        # Find symbol and amount among the groups
        symbol = None
        amount_str = None

        for g in groups:
            if g is None:
                continue
            if re.match(rf"^{self._CURRENCY_SYMBOLS}$|^{self._CURRENCY_CODES}$", g):
                symbol = g
            elif re.match(r"^\d", g):
                amount_str = g

        if amount_str is None:
            return {"symbol": symbol or "", "amount": 0}

        # Clean amount: remove thousands separators
        clean = re.sub(r"[\s]", "", amount_str)
        # Determine if last separator is decimal
        separators = re.findall(r"[,.]", clean)
        if separators:
            last_sep = separators[-1]
            # If the part after the last separator has 1-2 digits, it's decimal
            after_last = clean.split(last_sep)[-1]
            if len(after_last) <= 2:
                # Replace all separators except the last with nothing
                parts = clean.rsplit(last_sep, 1)
                integer_part = re.sub(r"[,.]", "", parts[0])
                clean = integer_part + "." + parts[1]
            else:
                clean = re.sub(r"[,.]", "", clean)

        try:
            amount = float(clean)
        except ValueError:
            amount = 0

        return {"symbol": symbol or "", "amount": amount}

    def _parse_date(self, match: re.Match, raw: str, config: LanguageConfig) -> dict:
        """Parse a date match into structured parts."""
        groups = [g for g in match.groups() if g is not None]

        # ISO format: 2024-01-15
        if "-" in raw and len(groups) == 3:
            return {"year": int(groups[0]), "month": int(groups[1]), "day": int(groups[2]),
                    "format": "ISO"}

        # Slash format: depends on locale
        if "/" in raw and len(groups) == 3:
            a, b, c = int(groups[0]), int(groups[1]), int(groups[2])
            if c < 100:
                c += 2000 if c < 50 else 1900
            if config.date_format == "MDY":
                return {"year": c, "month": a, "day": b, "format": "MDY"}
            else:
                return {"year": c, "month": b, "day": a, "format": "DMY"}

        # Month-name format: extract from raw text
        return {"raw": raw, "format": "named"}

    # Priority for overlap resolution: lower number = higher priority
    # More specific patterns should win over generic cardinal/year
    _CATEGORY_PRIORITY = {
        NumberCategory.CURRENCY: 0,
        NumberCategory.PERCENTAGE: 1,
        NumberCategory.DATE: 2,
        NumberCategory.TIME: 3,
        NumberCategory.ORDINAL: 4,
        NumberCategory.PHONE: 5,
        NumberCategory.DECIMAL: 6,
        NumberCategory.YEAR: 7,
        NumberCategory.CARDINAL: 8,
        NumberCategory.DIGIT_SEQUENCE: 9,
    }

    def _resolve_overlaps(self, spans: List[NumberSpan]) -> List[NumberSpan]:
        """
        Remove overlapping spans, keeping higher-priority (more specific)
        and longer spans.
        
        Priority order: currency > percentage > date > time > ordinal >
        phone > decimal > year > cardinal > digit_sequence
        
        When two spans overlap:
          1. Prefer the one with higher priority (lower priority number)
          2. If same priority, prefer the longer span
        """
        if not spans:
            return spans

        # Sort by: priority (most specific first), then longer spans first
        spans.sort(key=lambda s: (
            self._CATEGORY_PRIORITY.get(s.category, 99),
            -(s.end - s.start),  # longer spans first (negative for descending)
        ))

        result = []
        for span in spans:
            # Check if this span overlaps with any already accepted span
            overlaps = False
            for accepted in result:
                if span.start < accepted.end and span.end > accepted.start:
                    overlaps = True
                    break
            if not overlaps:
                result.append(span)

        return result
