"""Country values in one comparable form.

OpenSanctions gives countries as lowercase ISO 3166-1 alpha-2 codes ("ru", "ir"). OFAC writes them as English
names inside the Remarks column ("nationality Iran; citizen Iraq") or in the vessel-flag column. Customers give
whatever the application form had. normalize_country() maps all of these to a lowercase two-letter code when it
can, and otherwise keeps the casefolded name, so two values compare equal only when they mean the same country.

The name table below covers the country names that appear in the OFAC SDN file used here, plus a few common
alternative spellings. It is reference data, not a complete list: an unknown name is kept as text and simply
does not count as agreement or disagreement with a code.
"""

import re

from uwcopilot.kyc.normalize import strip_accents

COUNTRY_NAME_TO_ISO2: dict[str, str] = {
    "afghanistan": "af",
    "albania": "al",
    "algeria": "dz",
    "angola": "ao",
    "argentina": "ar",
    "armenia": "am",
    "australia": "au",
    "austria": "at",
    "azerbaijan": "az",
    "bahrain": "bh",
    "bangladesh": "bd",
    "barbados": "bb",
    "belarus": "by",
    "belgium": "be",
    "benin": "bj",
    "bolivia": "bo",
    "bosnia and herzegovina": "ba",
    "bosnia-herzegovina": "ba",
    "brazil": "br",
    "bulgaria": "bg",
    "burkina faso": "bf",
    "burma": "mm",
    "myanmar": "mm",
    "cabo verde": "cv",
    "cambodia": "kh",
    "cameroon": "cm",
    "canada": "ca",
    "cayman islands": "ky",
    "central african republic": "cf",
    "chad": "td",
    "chile": "cl",
    "china": "cn",
    "colombia": "co",
    "comoros": "km",
    "congo, democratic republic of the": "cd",
    "costa rica": "cr",
    "croatia": "hr",
    "cuba": "cu",
    "cyprus": "cy",
    "czech republic": "cz",
    "czechia": "cz",
    "denmark": "dk",
    "djibouti": "dj",
    "dominica": "dm",
    "dominican republic": "do",
    "ecuador": "ec",
    "egypt": "eg",
    "el salvador": "sv",
    "equatorial guinea": "gq",
    "eritrea": "er",
    "estonia": "ee",
    "ethiopia": "et",
    "fiji": "fj",
    "finland": "fi",
    "france": "fr",
    "gambia": "gm",
    "the gambia": "gm",
    "georgia": "ge",
    "germany": "de",
    "ghana": "gh",
    "greece": "gr",
    "guatemala": "gt",
    "guinea": "gn",
    "guyana": "gy",
    "haiti": "ht",
    "honduras": "hn",
    "hong kong": "hk",
    "hungary": "hu",
    "india": "in",
    "indonesia": "id",
    "iran": "ir",
    "iraq": "iq",
    "ireland": "ie",
    "israel": "il",
    "italy": "it",
    "japan": "jp",
    "jordan": "jo",
    "kazakhstan": "kz",
    "kenya": "ke",
    "korea, north": "kp",
    "north korea": "kp",
    "korea, south": "kr",
    "south korea": "kr",
    "kosovo": "xk",
    "kuwait": "kw",
    "kyrgyzstan": "kg",
    "latvia": "lv",
    "lebanon": "lb",
    "liberia": "lr",
    "libya": "ly",
    "liechtenstein": "li",
    "lithuania": "lt",
    "malaysia": "my",
    "maldives": "mv",
    "mali": "ml",
    "malta": "mt",
    "man, isle of": "im",
    "isle of man": "im",
    "mauritania": "mr",
    "mexico": "mx",
    "moldova": "md",
    "mongolia": "mn",
    "montenegro": "me",
    "morocco": "ma",
    "mozambique": "mz",
    "netherlands": "nl",
    "new zealand": "nz",
    "nicaragua": "ni",
    "niger": "ne",
    "nigeria": "ng",
    "north macedonia": "mk",
    "north macedonia, the republic of": "mk",
    "norway": "no",
    "oman": "om",
    "pakistan": "pk",
    "palestinian": "ps",
    "palestine": "ps",
    "panama": "pa",
    "paraguay": "py",
    "peru": "pe",
    "philippines": "ph",
    "poland": "pl",
    "portugal": "pt",
    "qatar": "qa",
    "romania": "ro",
    "russia": "ru",
    "russian federation": "ru",
    "rwanda": "rw",
    "saint kitts and nevis": "kn",
    "saudi arabia": "sa",
    "senegal": "sn",
    "serbia": "rs",
    "sierra leone": "sl",
    "singapore": "sg",
    "slovakia": "sk",
    "slovenia": "si",
    "somalia": "so",
    "south africa": "za",
    "south sudan": "ss",
    "spain": "es",
    "sri lanka": "lk",
    "sudan": "sd",
    "sweden": "se",
    "switzerland": "ch",
    "syria": "sy",
    "taiwan": "tw",
    "tajikistan": "tj",
    "tanzania": "tz",
    "thailand": "th",
    "togo": "tg",
    "trinidad and tobago": "tt",
    "tunisia": "tn",
    "turkey": "tr",
    "turkiye": "tr",
    "turkmenistan": "tm",
    "uganda": "ug",
    "ukraine": "ua",
    "united arab emirates": "ae",
    "united kingdom": "gb",
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
    "uzbekistan": "uz",
    "vanuatu": "vu",
    "venezuela": "ve",
    "vietnam": "vn",
    "virgin islands, british": "vg",
    "yemen": "ye",
    "zimbabwe": "zw",
}
# Codes that differ between conventions: the UK is "gb" in ISO 3166 but often written "uk".
CODE_ALIASES: dict[str, str] = {"uk": "gb"}
TWO_LETTER_CODE = re.compile(r"^[a-z]{2}$")


def normalize_country(value: str | None) -> str | None:
    """Return a lowercase ISO alpha-2 code when the value can be mapped, the casefolded name otherwise.

    None, empty strings and the OFAC null marker "-0-" return None (country unknown).
    """
    if value is None:
        return None
    text = " ".join(strip_accents(value).casefold().split())
    if not text or text == "-0-":
        return None
    text = CODE_ALIASES.get(text, text)
    if TWO_LETTER_CODE.match(text):
        return text
    return COUNTRY_NAME_TO_ISO2.get(text, text)


def normalize_countries(values: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Normalize several country values, dropping unknowns and duplicates while keeping their order."""
    result: list[str] = []
    for value in values:
        country = normalize_country(value)
        if country is not None and country not in result:
            result.append(country)
    return tuple(result)
