"""Country Flag and Team Crest Resolution Utilities for Zola Predict."""

from typing import Optional
import urllib.parse

# ISO 3166-1 alpha-2 flag codes (+ UK constituent countries) for flagcdn.com
COUNTRY_CODE_MAP = {
    # Africa
    "algeria": "dz",
    "angola": "ao",
    "benin": "bj",
    "botswana": "bw",
    "burkina faso": "bf",
    "burundi": "bi",
    "cameroon": "cm",
    "cape verde": "cv",
    "cape verde islands": "cv",
    "central african republic": "cf",
    "chad": "td",
    "comoros": "km",
    "congo": "cg",
    "congo dr": "cd",
    "dr congo": "cd",
    "democratic republic of the congo": "cd",
    "djibouti": "dj",
    "egypt": "eg",
    "equatorial guinea": "gq",
    "eritrea": "er",
    "eswatini": "sz",
    "swaziland": "sz",
    "ethiopia": "et",
    "gabon": "ga",
    "gambia": "gm",
    "ghana": "gh",
    "guinea": "gn",
    "guinea-bissau": "gw",
    "ivory coast": "ci",
    "cote d'ivoire": "ci",
    "côte d'ivoire": "ci",
    "kenya": "ke",
    "lesotho": "ls",
    "liberia": "lr",
    "libya": "ly",
    "madagascar": "mg",
    "malawi": "mw",
    "mali": "ml",
    "mauritania": "mr",
    "mauritius": "mu",
    "morocco": "ma",
    "mozambique": "mz",
    "namibia": "na",
    "niger": "ne",
    "nigeria": "ng",
    "rwanda": "rw",
    "sao tome and principe": "st",
    "senegal": "sn",
    "seychelles": "sc",
    "sierra leone": "sl",
    "somalia": "so",
    "south africa": "za",
    "south sudan": "ss",
    "sudan": "sd",
    "tanzania": "tz",
    "togo": "tg",
    "tunisia": "tn",
    "uganda": "ug",
    "zambia": "zm",
    "zimbabwe": "zw",

    # Europe
    "albania": "al",
    "andorra": "ad",
    "armenia": "am",
    "austria": "at",
    "azerbaijan": "az",
    "belarus": "by",
    "belgium": "be",
    "bosnia and herzegovina": "ba",
    "bosnia & herzegovina": "ba",
    "bulgaria": "bg",
    "croatia": "hr",
    "cyprus": "cy",
    "czech republic": "cz",
    "czechia": "cz",
    "denmark": "dk",
    "england": "gb-eng",
    "estonia": "ee",
    "faroe islands": "fo",
    "finland": "fi",
    "france": "fr",
    "georgia": "ge",
    "germany": "de",
    "gibraltar": "gi",
    "greece": "gr",
    "hungary": "hu",
    "iceland": "is",
    "ireland": "ie",
    "rep. of ireland": "ie",
    "republic of ireland": "ie",
    "israel": "il",
    "italy": "it",
    "kazakhstan": "kz",
    "kosovo": "xk",
    "latvia": "lv",
    "liechtenstein": "li",
    "lithuania": "lt",
    "luxembourg": "lu",
    "malta": "mt",
    "moldova": "md",
    "monaco": "mc",
    "montenegro": "me",
    "netherlands": "nl",
    "north macedonia": "mk",
    "fyr macedonia": "mk",
    "northern ireland": "gb-nir",
    "norway": "no",
    "poland": "pl",
    "portugal": "pt",
    "romania": "ro",
    "russia": "ru",
    "san marino": "sm",
    "scotland": "gb-sct",
    "serbia": "rs",
    "slovakia": "sk",
    "slovenia": "si",
    "spain": "es",
    "sweden": "se",
    "switzerland": "ch",
    "turkey": "tr",
    "türkiye": "tr",
    "ukraine": "ua",
    "united kingdom": "gb",
    "wales": "gb-wls",

    # Americas
    "argentina": "ar",
    "bolivia": "bo",
    "brazil": "br",
    "canada": "ca",
    "chile": "cl",
    "colombia": "co",
    "costa rica": "cr",
    "cuba": "cu",
    "dominican republic": "do",
    "ecuador": "ec",
    "el salvador": "sv",
    "guatemala": "gt",
    "haiti": "ht",
    "honduras": "hn",
    "jamaica": "jm",
    "mexico": "mx",
    "nicaragua": "ni",
    "panama": "pa",
    "paraguay": "py",
    "peru": "pe",
    "trinidad and tobago": "tt",
    "trinidad-and-tobago": "tt",
    "united states": "us",
    "usa": "us",
    "uruguay": "uy",
    "venezuela": "ve",

    # Asia & Oceania
    "australia": "au",
    "bahrain": "bh",
    "china": "cn",
    "india": "in",
    "indonesia": "id",
    "iran": "ir",
    "iraq": "iq",
    "japan": "jp",
    "jordan": "jo",
    "kuwait": "kw",
    "lebanon": "lb",
    "malaysia": "my",
    "new zealand": "nz",
    "north korea": "kp",
    "oman": "om",
    "pakistan": "pk",
    "palestine": "ps",
    "philippines": "ph",
    "qatar": "qa",
    "saudi arabia": "sa",
    "singapore": "sg",
    "south korea": "kr",
    "syria": "sy",
    "thailand": "th",
    "united arab emirates": "ae",
    "uae": "ae",
    "uzbekistan": "uz",
    "vietnam": "vn",
}


def get_country_flag_url(name: Optional[str]) -> Optional[str]:
    """Return a high-resolution flagcdn.com URL for a national team or country name."""
    if not name:
        return None
    normalized = name.lower().strip()
    code = COUNTRY_CODE_MAP.get(normalized)
    if code:
        return f"https://flagcdn.com/w80/{code}.png"
    return None


def get_team_monogram_svg(team_name: str) -> str:
    """Generate an inline SVG crest monogram for a team."""
    words = [w for w in (team_name or "").split() if w]
    if len(words) == 1:
        initials = words[0][:3].upper()
    elif len(words) == 2:
        initials = (words[0][:1] + words[1][:2]).upper()
    elif len(words) >= 3:
        initials = (words[0][:1] + words[1][:1] + words[2][:1]).upper()
    else:
        initials = "FC"

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">'
        f'<defs><linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">'
        f'<stop offset="0%" stop-color="#0284c7"/>'
        f'<stop offset="100%" stop-color="#0f172a"/>'
        f'</linearGradient></defs>'
        f'<rect width="64" height="64" rx="14" fill="url(#bg)" stroke="#38bdf8" stroke-width="2"/>'
        f'<text x="32" y="38" font-family="system-ui, -apple-system, sans-serif" font-size="20" font-weight="900" '
        f'fill="#ffffff" text-anchor="middle" letter-spacing="1">{initials}</text>'
        f'</svg>'
    )
    encoded = urllib.parse.quote(svg)
    return f"data:image/svg+xml;utf8,{encoded}"
