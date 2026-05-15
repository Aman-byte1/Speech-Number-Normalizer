import functools
import re
from typing import List, Optional

from num2words import num2words

from speech_number_norm.detector import NumberCategory, NumberSpan
from speech_number_norm.lang_config import (
    LanguageConfig,
    CurrencyInfo,
    get_config,
    CURRENCIES_GLOBAL,
)


@functools.lru_cache(maxsize=65536)
def _cached_num2words(number, lang: str = "en", to: str = "cardinal", **kwargs) -> str:
    try:
        return num2words(number, lang=lang, to=to)
    except (NotImplementedError, OverflowError, ValueError, TypeError):
        return ""


def _remove_hyphens(text: str) -> str:
    return text.replace("-", " ")


def _add_hyphens_to_tens(text: str) -> str:
    """
    Add hyphens between tens and units where appropriate.
    E.g., 'twenty one' -> 'twenty-one'
    """
    tens = [
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
        "vingt", "trente", "quarante", "cinquante", "soixante",  # French
        "zwanzig", "dreißig", "vierzig", "fünfzig", "sechzig", "siebzig", "achtzig", "neunzig",  # German
        "veinte", "treinta", "cuarenta", "cincuenta", "sesenta", "setenta", "ochenta", "noventa",  # Spanish
    ]
    for t in tens:
        pattern = re.compile(rf"\b({t})\s+(\w+)\b", re.IGNORECASE)
        text = pattern.sub(r"\1-\2", text)
    return text


class CandidateGenerator:
    """Generates candidate verbalizations for detected number spans."""

    def generate(
        self,
        span: NumberSpan,
        language: str = "en",
    ) -> List[str]:
        """Generate candidate verbalizations for a number span."""
        config = get_config(language)
        lang = config.num2words_lang

        dispatch = {
            NumberCategory.CARDINAL: self._gen_cardinal,
            NumberCategory.ORDINAL: self._gen_ordinal,
            NumberCategory.CURRENCY: self._gen_currency,
            NumberCategory.PERCENTAGE: self._gen_percentage,
            NumberCategory.DECIMAL: self._gen_decimal,
            NumberCategory.DATE: self._gen_date,
            NumberCategory.TIME: self._gen_time,
            NumberCategory.YEAR: self._gen_year,
            NumberCategory.PHONE: self._gen_phone,
            NumberCategory.DIGIT_SEQUENCE: self._gen_digit_sequence,
        }

        generator = dispatch.get(span.category, self._gen_cardinal)
        candidates = generator(span, lang, config)

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in candidates:
            c_clean = c.strip()
            c_lower = c_clean.lower()
            if c_lower and c_lower not in seen:
                seen.add(c_lower)
                unique.append(c_clean)

        # Fallback: if nothing was generated, return the raw text
        if not unique:
            unique = [span.raw_text]

        return unique

    # -------------------------------------------------------------------
    # Category-specific generators
    # -------------------------------------------------------------------

    def _gen_cardinal(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate cardinal number candidates."""
        if span.value is None:
            return [span.raw_text]

        num = int(span.value)
        candidates = []

        # Standard cardinal
        base = _cached_num2words(num, lang=lang, to="cardinal")
        if base:
            candidates.append(base)
            # Variant without hyphens
            no_hyphen = _remove_hyphens(base)
            if no_hyphen != base:
                candidates.append(no_hyphen)

        # For numbers like 1001, add "and" variant
        if config.and_word and num > 100 and num % 100 != 0:
            # Try to inject "and" before the last two digits
            # e.g., "one thousand one" -> "one thousand and one"
            without_and = _remove_hyphens(base)
            # This is language-specific, so we try a simple heuristic
            parts = without_and.rsplit(" ", 1)
            if len(parts) == 2:
                with_and = f"{parts[0]} {config.and_word} {parts[1]}"
                candidates.append(with_and)

        # Digit-by-digit for short numbers (e.g., "one oh one" for 101)
        if 100 <= num <= 9999:
            digit_by_digit = self._digit_by_digit(str(num), lang)
            if digit_by_digit:
                candidates.append(digit_by_digit)

        return candidates

    def _gen_ordinal(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate ordinal number candidates."""
        if span.value is None:
            return [span.raw_text]

        num = int(span.value)
        candidates = []

        base = _cached_num2words(num, lang=lang, to="ordinal")
        if base:
            candidates.append(base)
            no_hyphen = _remove_hyphens(base)
            if no_hyphen != base:
                candidates.append(no_hyphen)

        return candidates

    def _gen_currency(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate currency verbalization candidates."""
        parts = span.parts or {}
        amount = parts.get("amount", span.value or 0)
        symbol = parts.get("symbol", "")

        # Find currency info
        currency_info = self._find_currency(symbol, config)

        int_part = int(amount)
        frac_part = round((amount - int_part) * 100)

        candidates = []

        # Try num2words currency mode
        if currency_info:
            try:
                curr = num2words(
                    amount, lang=lang, to="currency",
                    currency=currency_info.code
                )
                if curr:
                    candidates.append(curr)
            except (NotImplementedError, ValueError, TypeError):
                pass

        # Manual construction: "thirty one dollars"
        num_words = _cached_num2words(int_part, lang=lang, to="cardinal")
        if num_words and currency_info:
            curr_name = (
                currency_info.name_singular if int_part == 1
                else currency_info.name_plural
            )
            candidates.append(f"{num_words} {curr_name}")

            if frac_part > 0:
                frac_words = _cached_num2words(frac_part, lang=lang, to="cardinal")
                sub_name = (
                    currency_info.subunit_singular if frac_part == 1
                    else currency_info.subunit_plural
                )
                if frac_words and sub_name:
                    candidates.append(
                        f"{num_words} {curr_name} {config.and_word} {frac_words} {sub_name}".strip()
                    )
                    candidates.append(
                        f"{num_words} {curr_name} {frac_words} {sub_name}"
                    )

        # Just the number without currency word
        if num_words:
            candidates.append(num_words)
            candidates.append(_remove_hyphens(num_words))

        return candidates

    def _gen_percentage(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate percentage candidates."""
        if span.value is None:
            return [span.raw_text]

        # Percentage words by language
        pct_words = {
            "en": "percent", "es": "por ciento", "fr": "pour cent",
            "de": "Prozent", "it": "per cento", "pt": "por cento",
            "ru": "процентов", "zh": "百分之", "ja": "パーセント",
            "ko": "퍼센트", "ar": "بالمائة", "hi": "प्रतिशत",
            "am": "በመቶ", "tr": "yüzde", "nl": "procent",
        }
        pct = pct_words.get(lang, pct_words.get(lang.split("_")[0], "percent"))

        candidates = []

        if span.value == int(span.value):
            num = int(span.value)
            base = _cached_num2words(num, lang=lang, to="cardinal")
        else:
            # Handle decimal percentages
            base = self._verbalize_decimal(span.value, lang, config)

        if base:
            # Chinese/Japanese: 百分之 comes before the number
            if lang in ("zh", "ja"):
                candidates.append(f"{pct}{base}")
            else:
                candidates.append(f"{base} {pct}")
                candidates.append(f"{_remove_hyphens(base)} {pct}")

        return candidates

    def _gen_decimal(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate decimal number candidates."""
        parts = span.parts or {}
        integer = parts.get("integer", "0")
        fraction = parts.get("fraction", "0")

        candidates = []

        # "three point one four"
        point_word = config.point_word or "point"
        int_words = _cached_num2words(int(integer), lang=lang, to="cardinal")

        # Fraction read digit-by-digit
        frac_digits = self._digit_by_digit(fraction, lang)
        if int_words and frac_digits:
            candidates.append(f"{int_words} {point_word} {frac_digits}")

        # Fraction as a whole number: "three point fourteen"
        frac_words = _cached_num2words(int(fraction), lang=lang, to="cardinal")
        if int_words and frac_words:
            candidates.append(f"{int_words} {point_word} {frac_words}")

        # Without hyphens
        for c in list(candidates):
            no_h = _remove_hyphens(c)
            if no_h != c:
                candidates.append(no_h)

        return candidates

    def _gen_date(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate date verbalization candidates."""
        parts = span.parts or {}
        fmt = parts.get("format", "")

        if fmt == "named":
            # For named dates, we can't easily parse — return raw text
            return [span.raw_text]

        year = parts.get("year")
        month = parts.get("month")
        day = parts.get("day")

        candidates = []

        # Month names (English for now — extensible via config)
        month_names = {
            "en": ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"],
            "es": ["", "enero", "febrero", "marzo", "abril", "mayo", "junio",
                   "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"],
            "fr": ["", "janvier", "février", "mars", "avril", "mai", "juin",
                   "juillet", "août", "septembre", "octobre", "novembre", "décembre"],
            "de": ["", "Januar", "Februar", "März", "April", "Mai", "Juni",
                   "Juli", "August", "September", "Oktober", "November", "Dezember"],
        }
        names = month_names.get(lang, month_names.get("en"))

        if month and 1 <= month <= 12 and day and year:
            month_name = names[month] if month < len(names) else str(month)
            day_ordinal = _cached_num2words(day, lang=lang, to="ordinal")
            day_cardinal = _cached_num2words(day, lang=lang, to="cardinal")
            year_words = self._verbalize_year(year, lang)

            # "January fifteenth, twenty twenty four"
            if day_ordinal:
                candidates.append(f"{month_name} {day_ordinal} {year_words}")
            # "January fifteen, twenty twenty four"
            if day_cardinal:
                candidates.append(f"{month_name} {day_cardinal} {year_words}")

            # "the fifteenth of January, twenty twenty four"
            if day_ordinal and lang == "en":
                candidates.append(f"the {day_ordinal} of {month_name} {year_words}")

            # DMY format: "fifteen January twenty twenty four"
            if day_cardinal:
                candidates.append(f"{day_cardinal} {month_name} {year_words}")

        return candidates

    def _gen_time(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate time verbalization candidates."""
        parts = span.parts or {}
        hour = parts.get("hour", 0)
        minute = parts.get("minute", 0)

        candidates = []
        h_words = _cached_num2words(hour, lang=lang, to="cardinal")
        m_words = _cached_num2words(minute, lang=lang, to="cardinal")

        if minute == 0:
            # "three o'clock"
            if lang == "en":
                candidates.append(f"{h_words} o'clock")
            candidates.append(h_words)
        elif minute < 10:
            # "three oh five"
            zero_word = _cached_num2words(0, lang=lang, to="cardinal")
            digit_word = _cached_num2words(minute, lang=lang, to="cardinal")
            if lang == "en":
                candidates.append(f"{h_words} oh {digit_word}")
            candidates.append(f"{h_words} {zero_word} {digit_word}")
            candidates.append(f"{h_words} {m_words}")
        else:
            # "three thirty"
            candidates.append(f"{h_words} {m_words}")

        # 24-hour style: "fourteen hundred" for 14:00
        if hour >= 12 and minute == 0 and lang == "en":
            candidates.append(f"{_cached_num2words(hour, lang=lang)} hundred")

        return candidates

    def _gen_year(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate year verbalization candidates."""
        if span.value is None:
            return [span.raw_text]

        year = int(span.value)
        candidates = []

        # Year-style verbalization
        year_verb = self._verbalize_year(year, lang)
        if year_verb:
            candidates.append(year_verb)
            candidates.append(_remove_hyphens(year_verb))

        # Full cardinal
        cardinal = _cached_num2words(year, lang=lang, to="cardinal")
        if cardinal and cardinal != year_verb:
            candidates.append(cardinal)
            candidates.append(_remove_hyphens(cardinal))

        # num2words year mode (some languages support it)
        try:
            yr = num2words(year, lang=lang, to="year")
            if yr and yr not in candidates:
                candidates.append(yr)
        except (NotImplementedError, ValueError, TypeError):
            pass

        return candidates

    def _gen_phone(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate phone number / digit sequence candidates."""
        parts = span.parts or {}
        digits = parts.get("digits", re.sub(r"\D", "", span.raw_text))
        return [self._digit_by_digit(digits, lang)]

    def _gen_digit_sequence(
        self, span: NumberSpan, lang: str, config: LanguageConfig
    ) -> List[str]:
        """Generate digit-by-digit candidates."""
        digits = re.sub(r"\D", "", span.raw_text)
        return [self._digit_by_digit(digits, lang)]

    # -------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------

    def _digit_by_digit(self, digits: str, lang: str) -> str:
        """Convert a string of digits to word-by-word representation."""
        words = []
        for d in digits:
            w = _cached_num2words(int(d), lang=lang, to="cardinal")
            if w:
                words.append(w)
        return " ".join(words)

    def _verbalize_year(self, year: int, lang: str) -> str:
        """
        Verbalize a year in the natural style.
        English: 1984 -> "nineteen eighty four", 2024 -> "twenty twenty four"
        Other languages: use num2words cardinal as fallback.
        """
        if lang in ("en", "en_GB", "en_US"):
            if 1000 <= year <= 1999:
                first = year // 100
                second = year % 100
                first_w = _cached_num2words(first, lang=lang, to="cardinal")
                if second == 0:
                    return f"{first_w} hundred"
                second_w = _cached_num2words(second, lang=lang, to="cardinal")
                return f"{first_w} {second_w}"
            elif 2000 <= year <= 2009:
                return _cached_num2words(year, lang=lang, to="cardinal")
            elif 2010 <= year <= 2099:
                first = year // 100
                second = year % 100
                first_w = _cached_num2words(first, lang=lang, to="cardinal")
                second_w = _cached_num2words(second, lang=lang, to="cardinal")
                return f"{first_w} {second_w}"

        # Fallback: use cardinal
        return _cached_num2words(year, lang=lang, to="cardinal")

    def _verbalize_decimal(self, value: float, lang: str, config: LanguageConfig) -> str:
        """Verbalize a decimal number."""
        str_val = str(value)
        if "." in str_val:
            integer, fraction = str_val.split(".", 1)
            int_words = _cached_num2words(int(integer), lang=lang, to="cardinal")
            frac_digits = self._digit_by_digit(fraction, lang)
            point_word = config.point_word or "point"
            return f"{int_words} {point_word} {frac_digits}"
        return _cached_num2words(int(value), lang=lang, to="cardinal")

    def _find_currency(
        self, symbol: str, config: LanguageConfig
    ) -> Optional[CurrencyInfo]:
        """Find currency info by symbol or code."""
        # Check language-specific currencies first
        for curr in config.currencies:
            if curr.symbol == symbol or curr.code == symbol:
                return curr
        # Fall back to global
        if symbol in CURRENCIES_GLOBAL:
            return CURRENCIES_GLOBAL[symbol]
        # Try matching by code
        for curr in CURRENCIES_GLOBAL.values():
            if curr.code == symbol:
                return curr
        return None
