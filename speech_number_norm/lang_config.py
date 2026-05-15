"""
Language-specific configuration for number normalization.

Defines per-language settings for currency symbols, date formats,
decimal/thousands separators, and other locale-specific patterns.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class CurrencyInfo:
    """Info about a currency symbol."""
    symbol: str
    code: str           # ISO 4217 code used by num2words
    name_singular: str  # e.g. "dollar"
    name_plural: str    # e.g. "dollars"
    subunit_singular: str = ""  # e.g. "cent"
    subunit_plural: str = ""    # e.g. "cents"
    symbol_before: bool = True  # True if symbol comes before number ($100 vs 100€)


@dataclass
class LanguageConfig:
    """Per-language configuration for number normalization."""
    code: str                        # ISO 639-1 code
    name: str                        # Human-readable name
    num2words_lang: str              # Code used by num2words library
    decimal_separator: str = "."     # "." or ","
    thousands_separator: str = ","   # "," or "." or " "
    date_format: str = "MDY"         # MDY, DMY, or YMD
    currencies: List[CurrencyInfo] = field(default_factory=list)
    ordinal_suffixes: List[str] = field(default_factory=list)  # e.g. ["st", "nd", "rd", "th"]
    and_word: str = ""               # Word for "and" in number speech (e.g. "and" in English)
    point_word: str = ""             # Word for decimal point
    # MMS language code for forced alignment (uroman-based, usually same as ISO)
    mms_lang: str = ""


# ---------------------------------------------------------------------------
# Currency database (most common currencies by symbol)
# ---------------------------------------------------------------------------

CURRENCIES_GLOBAL: Dict[str, CurrencyInfo] = {
    "$": CurrencyInfo("$", "USD", "dollar", "dollars", "cent", "cents", True),
    "€": CurrencyInfo("€", "EUR", "euro", "euros", "cent", "cents", True),
    "£": CurrencyInfo("£", "GBP", "pound", "pounds", "penny", "pence", True),
    "¥": CurrencyInfo("¥", "JPY", "yen", "yen", "", "", True),
    "₹": CurrencyInfo("₹", "INR", "rupee", "rupees", "paisa", "paise", True),
    "₽": CurrencyInfo("₽", "RUB", "ruble", "rubles", "kopek", "kopeks", False),
    "₩": CurrencyInfo("₩", "KRW", "won", "won", "", "", True),
    "₪": CurrencyInfo("₪", "ILS", "shekel", "shekels", "agora", "agorot", True),
    "฿": CurrencyInfo("฿", "THB", "baht", "baht", "satang", "satang", True),
    "R": CurrencyInfo("R", "ZAR", "rand", "rand", "cent", "cents", True),
    "kr": CurrencyInfo("kr", "SEK", "krona", "kronor", "öre", "öre", False),
    "CHF": CurrencyInfo("CHF", "CHF", "franc", "francs", "centime", "centimes", True),
    "ETB": CurrencyInfo("ETB", "ETB", "birr", "birr", "santim", "santim", True),
    "Br": CurrencyInfo("Br", "ETB", "birr", "birr", "santim", "santim", True),
}


# ---------------------------------------------------------------------------
# Language configurations
# ---------------------------------------------------------------------------

LANGUAGE_CONFIGS: Dict[str, LanguageConfig] = {
    "en": LanguageConfig(
        code="en", name="English", num2words_lang="en",
        decimal_separator=".", thousands_separator=",",
        date_format="MDY",
        ordinal_suffixes=["st", "nd", "rd", "th"],
        and_word="and", point_word="point",
        mms_lang="eng",
        currencies=[CURRENCIES_GLOBAL["$"], CURRENCIES_GLOBAL["£"], CURRENCIES_GLOBAL["€"]],
    ),
    "es": LanguageConfig(
        code="es", name="Spanish", num2words_lang="es",
        decimal_separator=",", thousands_separator=".",
        date_format="DMY",
        ordinal_suffixes=["º", "ª", "er"],
        and_word="y", point_word="coma",
        mms_lang="spa",
        currencies=[CURRENCIES_GLOBAL["€"], CURRENCIES_GLOBAL["$"]],
    ),
    "fr": LanguageConfig(
        code="fr", name="French", num2words_lang="fr",
        decimal_separator=",", thousands_separator=" ",
        date_format="DMY",
        ordinal_suffixes=["er", "ère", "e", "ème"],
        and_word="et", point_word="virgule",
        mms_lang="fra",
        currencies=[CURRENCIES_GLOBAL["€"]],
    ),
    "de": LanguageConfig(
        code="de", name="German", num2words_lang="de",
        decimal_separator=",", thousands_separator=".",
        date_format="DMY",
        ordinal_suffixes=[".", "te", "ter", "tes"],
        and_word="und", point_word="komma",
        mms_lang="deu",
        currencies=[CURRENCIES_GLOBAL["€"]],
    ),
    "it": LanguageConfig(
        code="it", name="Italian", num2words_lang="it",
        decimal_separator=",", thousands_separator=".",
        date_format="DMY",
        ordinal_suffixes=["º", "ª", "°"],
        and_word="e", point_word="virgola",
        mms_lang="ita",
        currencies=[CURRENCIES_GLOBAL["€"]],
    ),
    "pt": LanguageConfig(
        code="pt", name="Portuguese", num2words_lang="pt",
        decimal_separator=",", thousands_separator=".",
        date_format="DMY",
        ordinal_suffixes=["º", "ª"],
        and_word="e", point_word="vírgula",
        mms_lang="por",
        currencies=[CURRENCIES_GLOBAL["€"], CURRENCIES_GLOBAL["$"]],
    ),
    "ru": LanguageConfig(
        code="ru", name="Russian", num2words_lang="ru",
        decimal_separator=",", thousands_separator=" ",
        date_format="DMY",
        ordinal_suffixes=[],
        and_word="и", point_word="запятая",
        mms_lang="rus",
        currencies=[CURRENCIES_GLOBAL["₽"]],
    ),
    "zh": LanguageConfig(
        code="zh", name="Chinese", num2words_lang="zh",
        decimal_separator=".", thousands_separator=",",
        date_format="YMD",
        ordinal_suffixes=[],
        and_word="", point_word="点",
        mms_lang="cmn",
        currencies=[CURRENCIES_GLOBAL["¥"]],
    ),
    "ja": LanguageConfig(
        code="ja", name="Japanese", num2words_lang="ja",
        decimal_separator=".", thousands_separator=",",
        date_format="YMD",
        ordinal_suffixes=[],
        and_word="", point_word="点",
        mms_lang="jpn",
        currencies=[CURRENCIES_GLOBAL["¥"]],
    ),
    "ko": LanguageConfig(
        code="ko", name="Korean", num2words_lang="ko",
        decimal_separator=".", thousands_separator=",",
        date_format="YMD",
        ordinal_suffixes=[],
        and_word="", point_word="점",
        mms_lang="kor",
        currencies=[CURRENCIES_GLOBAL["₩"]],
    ),
    "ar": LanguageConfig(
        code="ar", name="Arabic", num2words_lang="ar",
        decimal_separator=".", thousands_separator=",",
        date_format="DMY",
        ordinal_suffixes=[],
        and_word="و", point_word="فاصلة",
        mms_lang="arb",
        currencies=[],
    ),
    "hi": LanguageConfig(
        code="hi", name="Hindi", num2words_lang="hi",
        decimal_separator=".", thousands_separator=",",
        date_format="DMY",
        ordinal_suffixes=[],
        and_word="और", point_word="दशमलव",
        mms_lang="hin",
        currencies=[CURRENCIES_GLOBAL["₹"]],
    ),
    "am": LanguageConfig(
        code="am", name="Amharic", num2words_lang="am",
        decimal_separator=".", thousands_separator=",",
        date_format="DMY",
        ordinal_suffixes=[],
        and_word="እና", point_word="ነጥብ",
        mms_lang="amh",
        currencies=[CURRENCIES_GLOBAL.get("ETB", CURRENCIES_GLOBAL["Br"])],
    ),
    "tr": LanguageConfig(
        code="tr", name="Turkish", num2words_lang="tr",
        decimal_separator=",", thousands_separator=".",
        date_format="DMY",
        ordinal_suffixes=[".", "inci", "ıncı", "üncü", "uncu"],
        and_word="ve", point_word="virgül",
        mms_lang="tur",
        currencies=[],
    ),
    "pl": LanguageConfig(
        code="pl", name="Polish", num2words_lang="pl",
        decimal_separator=",", thousands_separator=" ",
        date_format="DMY",
        ordinal_suffixes=["."],
        and_word="i", point_word="przecinek",
        mms_lang="pol",
        currencies=[],
    ),
    "nl": LanguageConfig(
        code="nl", name="Dutch", num2words_lang="nl",
        decimal_separator=",", thousands_separator=".",
        date_format="DMY",
        ordinal_suffixes=["e", "de", "ste"],
        and_word="en", point_word="komma",
        mms_lang="nld",
        currencies=[CURRENCIES_GLOBAL["€"]],
    ),
    "sv": LanguageConfig(
        code="sv", name="Swedish", num2words_lang="sv",
        decimal_separator=",", thousands_separator=" ",
        date_format="YMD",
        ordinal_suffixes=["a", "e"],
        and_word="och", point_word="komma",
        mms_lang="swe",
        currencies=[CURRENCIES_GLOBAL["kr"]],
    ),
    "uk": LanguageConfig(
        code="uk", name="Ukrainian", num2words_lang="uk",
        decimal_separator=",", thousands_separator=" ",
        date_format="DMY",
        ordinal_suffixes=[],
        and_word="і", point_word="кома",
        mms_lang="ukr",
        currencies=[],
    ),
    "vi": LanguageConfig(
        code="vi", name="Vietnamese", num2words_lang="vi",
        decimal_separator=",", thousands_separator=".",
        date_format="DMY",
        ordinal_suffixes=[],
        and_word="và", point_word="phẩy",
        mms_lang="vie",
        currencies=[],
    ),
    "th": LanguageConfig(
        code="th", name="Thai", num2words_lang="th",
        decimal_separator=".", thousands_separator=",",
        date_format="DMY",
        ordinal_suffixes=[],
        and_word="", point_word="จุด",
        mms_lang="tha",
        currencies=[CURRENCIES_GLOBAL["฿"]],
    ),
    "id": LanguageConfig(
        code="id", name="Indonesian", num2words_lang="id",
        decimal_separator=",", thousands_separator=".",
        date_format="DMY",
        ordinal_suffixes=["ke-"],
        and_word="dan", point_word="koma",
        mms_lang="ind",
        currencies=[],
    ),
    "he": LanguageConfig(
        code="he", name="Hebrew", num2words_lang="he",
        decimal_separator=".", thousands_separator=",",
        date_format="DMY",
        ordinal_suffixes=[],
        and_word="ו", point_word="נקודה",
        mms_lang="heb",
        currencies=[CURRENCIES_GLOBAL["₪"]],
    ),
}


def get_config(lang: str) -> LanguageConfig:
    """
    Get language configuration, falling back to English defaults
    if the language is not explicitly configured.
    """
    if lang in LANGUAGE_CONFIGS:
        return LANGUAGE_CONFIGS[lang]

    # Try to find a base language match (e.g. "en_US" -> "en")
    base = lang.split("_")[0].split("-")[0].lower()
    if base in LANGUAGE_CONFIGS:
        return LANGUAGE_CONFIGS[base]

    # Return a generic config that uses the lang code directly with num2words
    return LanguageConfig(
        code=lang,
        name=lang,
        num2words_lang=lang,
        mms_lang=lang,
    )


def supported_languages() -> List[str]:
    """Return list of explicitly supported language codes."""
    return sorted(LANGUAGE_CONFIGS.keys())
