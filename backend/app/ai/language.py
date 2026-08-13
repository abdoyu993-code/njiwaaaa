from __future__ import annotations

import re
import unicodedata


"""
Détection de langue et normalisation vers le français.

Le classifieur d'intentions et le corpus documentaire sont en
français. Pour comprendre l'arabe, la darija (en alphabet arabe)
et l'arabizi, la question est traduite terme à terme vers un
français « de recherche » qui alimente la classification et le
RAG — la question d'origine reste utilisée pour la réponse.
"""


LANGUAGE_FRENCH = "fr"
LANGUAGE_ARABIC = "ar"
LANGUAGE_DARIJA = "darija"
LANGUAGE_ARABIZI = "arabizi"

# Langue dans laquelle la réponse doit être rédigée.
RESPONSE_LANGUAGES = {
    LANGUAGE_FRENCH: "français",
    LANGUAGE_ARABIC: "arabe standard",
    LANGUAGE_DARIJA: "darija marocaine écrite en alphabet arabe",
    LANGUAGE_ARABIZI: (
        "darija marocaine écrite en lettres latines (arabizi)"
    ),
}

_ARABIC_BLOCK = re.compile(r"[؀-ۿ]")

# Mots typiquement darija (alphabet arabe) : ils distinguent la
# darija de l'arabe standard.
DARIJA_ARABIC_MARKERS = (
    "بغيت",
    "بغينا",
    "كيفاش",
    "فين",
    "شحال",
    "واش",
    "خاصني",
    "خصني",
    "ديالي",
    "ديال",
    "شنو",
    "علاش",
    "دابا",
    "مزال",
    "بزاف",
    "غادي",
    "كنبغي",
    "ماعندي",
)

# Marqueurs arabizi (darija en lettres latines).
ARABIZI_MARKERS = (
    "bghit",
    "bgit",
    "chno",
    "chnou",
    "kifach",
    "kifash",
    "khassni",
    "khasni",
    "dyali",
    "dyal",
    "3andi",
    "3la",
    "wach",
    "wa7ed",
    "chhal",
    "ch7al",
    "fin ",
    "ndir",
    "wra9",
    "wraq",
    "salat",
    "tal7at",
    "daz",
    "mchat",
    "bzaf",
    "daba",
    "mzal",
    "3lach",
)

# Traduction terme à terme vers le français de recherche.
# Arabe standard et darija en alphabet arabe.
ARABIC_LEXICON: dict[str, str] = {
    "رخصة السياقة": "permis de conduire",
    "رخصة السياقه": "permis de conduire",
    "رخصة القيادة": "permis de conduire",
    "الرخصة": "permis de conduire",
    "رخصة": "permis de conduire",
    "بيرمي": "permis de conduire",
    "البطاقة الرمادية": "carte grise",
    "البطاقه الرماديه": "carte grise",
    "الكارت غريز": "carte grise",
    "بطاقة التسجيل": "certificat immatriculation carte grise",
    "شهادة التسجيل": "certificat immatriculation carte grise",
    "تسجيل السيارة": "immatriculation véhicule",
    "ترقيم": "immatriculation",
    "تفويت": "mutation changement propriétaire",
    "بيع السيارة": "vente véhicule mutation",
    "شراء سيارة": "achat véhicule mutation",
    "الفحص التقني": "visite technique contrôle technique",
    "الفحص التقنى": "visite technique contrôle technique",
    "المراقبة التقنية": "visite technique contrôle technique",
    "موعد": "rendez-vous",
    "المخالفة": "infraction contravention",
    "مخالفة": "infraction contravention",
    "الغرامة": "amende",
    "غرامة": "amende",
    "الذعيرة": "amende contravention",
    "أداء الغرامة": "payer amende",
    "خلاص الغرامة": "payer amende",
    "الطعن": "réclamation contestation",
    "شكاية": "réclamation",
    "الشكاية": "réclamation",
    "حادثة": "accident",
    "حادث": "accident",
    "حادثة سير": "accident de la route",
    "التأمين": "assurance",
    "ضاعت": "perdu perte",
    "ضايعة": "perdu perte",
    "ضاع": "perdu perte",
    "سرقة": "vol volé",
    "مسروقة": "volé",
    "نسخة": "duplicata",
    "دوبليكة": "duplicata",
    "الوثائق": "documents nécessaires",
    "وثائق": "documents nécessaires",
    "الأوراق": "documents nécessaires",
    "أوراق": "documents nécessaires",
    "مطبوع": "formulaire",
    "استمارة": "formulaire",
    "الاستمارة": "formulaire",
    "كيفاش": "comment",
    "كيف": "comment",
    "فين": "où",
    "أين": "où",
    "شحال": "combien",
    "كم": "combien",
    "الثمن": "prix tarif",
    "الأجل": "délai",
    "المدة": "délai",
    "بغيت": "je veux",
    "أريد": "je veux",
    "خاصني": "j'ai besoin de",
    "السيارة": "véhicule voiture",
    "سيارة": "véhicule voiture",
    "الطوموبيل": "véhicule voiture",
    "دراجة": "moto deux-roues",
    "الخوذة": "casque",
    "حزام السلامة": "ceinture de sécurité",
    "الهاتف": "téléphone",
    "السرعة": "vitesse",
    "النقط": "points permis",
    "التواصل": "contact",
    "الاتصال": "contact",
}

# Arabizi et darija translittérée.
ARABIZI_LEXICON: dict[str, str] = {
    "permis dyali": "mon permis de conduire",
    "carte grise dyali": "ma carte grise",
    "bghit": "je veux",
    "bgit": "je veux",
    "khassni": "j'ai besoin de",
    "khasni": "j'ai besoin de",
    "kifach": "comment",
    "kifash": "comment",
    "chno ndir": "que dois-je faire",
    "chnou ndir": "que dois-je faire",
    "chno": "quoi",
    "chnou": "quoi",
    "3lach": "pourquoi",
    "wach": "est-ce que",
    "chhal": "combien",
    "ch7al": "combien",
    "wra9": "documents",
    "wraq": "documents",
    "salat": "expiré",
    "salate": "expiré",
    "tal7at": "perdu",
    "twadar": "perdu",
    "tsra9": "volé",
    "sra9": "volé",
    "njdad": "renouveler",
    "jdid": "nouveau",
    "dyal": "de",
    "dyali": "mon",
    "daba": "maintenant",
    "mzal": "encore",
    "bzaf": "beaucoup",
    "flous": "frais paiement",
    "khlas": "payer",
    "n5alas": "payer",
    "nkhalas": "payer",
    "mo5alafa": "contravention infraction",
    "mokhalafa": "contravention infraction",
    "hadita": "accident",
    "ksida": "accident",
    "visite technique": "visite technique",
    "l7adid": "",
}


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())

    return "".join(
        character
        for character in decomposed
        if unicodedata.category(character) != "Mn"
    )


def detect_language(text: str) -> str:
    """
    Reconnaît la langue d'une question.

    Retourne `fr`, `ar`, `darija` (alphabet arabe) ou `arabizi`.
    """
    if not text or not text.strip():
        return LANGUAGE_FRENCH

    arabic_characters = len(_ARABIC_BLOCK.findall(text))

    if arabic_characters >= 2:
        if any(
            marker in text
            for marker in DARIJA_ARABIC_MARKERS
        ):
            return LANGUAGE_DARIJA

        return LANGUAGE_ARABIC

    normalized = _strip_accents(text)
    padded = f" {normalized} "

    arabizi_hits = sum(
        1
        for marker in ARABIZI_MARKERS
        if marker in padded
    )

    # Les chiffres utilisés comme lettres (3, 7, 9) sont un
    # indice fort d'arabizi.
    letter_digits = len(
        re.findall(r"[a-z][3579][a-z]", normalized)
    )

    if arabizi_hits >= 1 or letter_digits >= 1:
        return LANGUAGE_ARABIZI

    return LANGUAGE_FRENCH


def to_search_query(
    text: str,
    language: str | None = None,
) -> str:
    """
    Traduit la question vers un français de recherche.

    Le texte français est renvoyé tel quel ; les autres langues
    sont converties terme à terme à partir des lexiques.
    """
    language = language or detect_language(text)

    if language == LANGUAGE_FRENCH:
        return text

    if language in {LANGUAGE_ARABIC, LANGUAGE_DARIJA}:
        translated = text

        # Les expressions les plus longues d'abord, afin que
        # « رخصة السياقة » ne soit pas coupée en deux.
        for term in sorted(
            ARABIC_LEXICON,
            key=len,
            reverse=True,
        ):
            if term in translated:
                translated = translated.replace(
                    term,
                    f" {ARABIC_LEXICON[term]} ",
                )

        # Ce qui reste en arabe n'apporte rien à la recherche.
        translated = _ARABIC_BLOCK.sub(" ", translated)
        cleaned = " ".join(translated.split())

        return cleaned or text

    normalized = _strip_accents(text)
    translated = f" {normalized} "

    for term in sorted(
        ARABIZI_LEXICON,
        key=len,
        reverse=True,
    ):
        if term in translated:
            translated = translated.replace(
                term,
                f" {ARABIZI_LEXICON[term]} ",
            )

    return " ".join(translated.split()) or text


def response_instruction(language: str) -> str:
    """Consigne de langue ajoutée au prompt de génération."""
    label = RESPONSE_LANGUAGES.get(
        language,
        RESPONSE_LANGUAGES[LANGUAGE_FRENCH],
    )

    if language == LANGUAGE_FRENCH:
        return "Réponds en français."

    return (
        f"Réponds en {label}, car c'est la langue de la "
        "question. Conserve tels quels, sans les traduire : les "
        "adresses internet, les noms officiels des formulaires "
        "(Formule I, Formule BI…), les noms d'organismes "
        "(NARSA, Al Barid Bank, Barid Cash) et les références "
        "réglementaires."
    )


# Message affiché lorsque la réponse documentaire ne peut pas
# être traduite (génération indisponible).
UNTRANSLATED_NOTICE = {
    LANGUAGE_ARABIC: (
        "الوثائق الرسمية متوفرة بالفرنسية. إليك المعلومات "
        "المطلوبة :"
    ),
    LANGUAGE_DARIJA: (
        "الوثائق الرسمية كاينة بالفرنسية. ها المعلومات "
        "اللي طلبتي :"
    ),
    LANGUAGE_ARABIZI: (
        "Les documents officiels sont en français. Voici "
        "l'information demandée :"
    ),
}


def untranslated_notice(language: str) -> str:
    return UNTRANSLATED_NOTICE.get(language, "")
