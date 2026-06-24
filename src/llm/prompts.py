"""Prompt templates for LLM-based insurance policy extraction.

Version-controlled prompts enable reproducibility across runs.
All prompts are in Dutch since the source documents are Dutch.
"""

from __future__ import annotations

PROMPT_VERSION = "v2.0"

# ── Coverage / exclusion theme taxonomy ──────────────────────────────────
# Learned from corpus analysis; used for icon mapping and clustering.

COVERAGE_THEMES = [
    "liability",            # Aansprakelijkheid / schade aan anderen
    "own_damage",           # Eigen schade / botsing / aanrijding
    "fire",                 # Brand / zelfontbranding
    "theft",                # Diefstal / inbraak / joyriding
    "storm_weather",        # Storm / hagel / natuurgeweld
    "glass",                # Ruitschade
    "vandalism",            # Vandalisme
    "breakdown",            # Pech / motorstoring
    "assistance",           # Hulpverlening / berging / vervoer
    "replacement_vehicle",  # Vervangende auto
    "passengers",           # Inzittenden / letsel passagiers
    "legal",                # Rechtsbijstand / juridische hulp
    "accessories",          # Extra's / accessoires / navigatie
    "valuation",            # Waardebepaling / dagwaarde / nieuwwaarde
    "animal_damage",        # Schade door dieren
    "water_damage",         # Waterschade / overstroming
    "transport",            # Schade tijdens transport
    "other_coverage",       # Overige dekking
]

EXCLUSION_THEMES = [
    "driver_restrictions",  # Alcohol / drugs / rijbewijs / bevoegdheid
    "usage_restrictions",   # Zakelijk gebruik / geen toestemming
    "behavior",             # Opzet / roekeloosheid
    "technical",            # Onderhoud / slijtage / overbelasting
    "events",               # Wedstrijd / circuit / evenementen
    "crime_fraud",          # Criminele activiteiten / fraude
    "extreme_events",       # Atoomkernreacties / terrorisme / molest
    "rental_commercial",    # Verhuur / lease aan derden
    "other_exclusion",      # Overige uitsluiting
]

DUTY_THEMES = [
    "police_reporting",     # Melding bij politie
    "damage_reporting",     # Schade melden bij verzekeraar
    "change_reporting",     # Wijzigingen doorgeven
    "cooperation",          # Meewerken aan onderzoek
    "damage_mitigation",    # Schade beperken
    "alarmcentrale",        # Contact alarmcentrale
    "documentation",        # Bewijsstukken / formulieren
]

ALL_THEMES = COVERAGE_THEMES + EXCLUSION_THEMES + DUTY_THEMES

# ── Pass 1: Section-level structured extraction ──────────────────────────

SYSTEM_PROMPT_EXTRACTION = """\
Je bent een expert in Nederlandse autoverzekeringen. Je analyseert secties \
uit verzekeringspolissen en extraheert gestructureerde informatie.

TAAK:
Lees de sectie hieronder en extraheer ALLE relevante beleidsstatements. \
Elk statement moet apart worden geretourneerd.

REGELS:
- Extraheer alleen wat EXPLICIET in de tekst staat.
- Infereer GEEN bredere dekking dan beschreven.
- Gebruik EXACTE quotes uit de brontekst (kopieer letterlijk).
- Verwijs naar de [clause_id=...] markers die in de tekst staan.
- Negeer inhoudsopgaven, paginanummers en navigatietekst.
- Markeer definities apart (item_type="definition").
- Markeer administratieve/claimproces tekst apart (item_type="admin").
- KOLOM-LABELS: Als een clausule het label [column=insured] heeft, is het item \
"covered" (tenzij de tekst expliciet zegt dat iets NIET verzekerd is). \
Als het label [column=not_insured] heeft, is het item "not_covered". \
Deze labels komen uit de tabelstructuur van het brondocument en zijn betrouwbaar.
- HINT-LABELS: Als een clausule het label [hint=not_covered] heeft, is het \
waarschijnlijk een uitsluiting. Als [hint=covered], is het waarschijnlijk dekking. \
Gebruik de hint als leidraad, maar de tekst heeft voorrang bij tegenstrijdigheid.

ITEM TYPES:
- covered: De klant IS verzekerd voor iets (dekking, vergoeding, hulp)
- not_covered: Een uitsluiting (iets dat NIET verzekerd is)
- condition: Een voorwaarde waaraan voldaan moet worden voor dekking
- limit: Een maximumbedrag, maximale tijd, of maximum aantal
- deadline: Een termijn waarbinnen iets moet gebeuren
- notification_duty: Een MELDPLICHT (iets melden bij politie, verzekeraar, etc.)
- claim_obligation: Een VERPLICHTING bij schadeafhandeling (meewerken, bewijs, etc.)
- obligation: Een andere plicht van de klant
- definition: Een uitleg/definitie van een begrip
- admin: Administratieve of interne procestekst

MODULES:
- wa: Wettelijke Aansprakelijkheid (schade aan anderen)
- beperkt_casco: Beperkt Casco (brand, diefstal, storm, ruitschade)
- all_risk: All Risk / Volledig Casco (alle schade aan eigen auto)
- pechhulp: Pechhulp / Hulpdiensten bij pech
- rechtsbijstand: Rechtsbijstand / Juridische hulp
- inzittenden: Inzittendenverzekering (letsel passagiers)
- general: Algemene bepalingen, niet specifiek voor één module
Gebruik het sectiepad (bijv. "Beperkt Casco > Dit is verzekerd") om de module \
te bepalen, tenzij de tekst expliciet naar een andere module verwijst.

HEADLINE:
- Voor item_type covered: geef een korte klantvriendelijke samenvatting \
(max 10 woorden). Focus op WAT er gedekt is.
- Voor item_type not_covered: geef een korte samenvatting van de uitsluiting \
(max 10 woorden).
- Voor notification_duty/claim_obligation: geef een korte beschrijving \
van de plicht (max 10 woorden).
- Voor andere types: geef null.

EXCLUSION_SCOPE (alleen voor not_covered):
- Lijst van modules waarop deze uitsluiting van toepassing is.
- Bijv. ["wa", "beperkt_casco", "all_risk"] voor algemene uitsluitingen.
- Bijv. ["beperkt_casco"] voor casco-specifieke uitsluitingen.
- Als onduidelijk: ["general"].

THEME:
Wijs elk item een theme toe uit de onderstaande lijst:

Coverage themes: liability, own_damage, fire, theft, storm_weather, glass, \
vandalism, breakdown, assistance, replacement_vehicle, passengers, legal, \
accessories, valuation, animal_damage, water_damage, transport, other_coverage

Exclusion themes: driver_restrictions, usage_restrictions, behavior, \
technical, events, crime_fraud, extreme_events, rental_commercial, \
other_exclusion

Duty themes: police_reporting, damage_reporting, change_reporting, \
cooperation, damage_mitigation, alarmcentrale, documentation

IMPORTANCE (1-5):
- 5: Kerndekkingen, grote bedragen, kritieke uitsluitingen
- 4: Belangrijke aanvullende dekkingen of voorwaarden
- 3: Nuttige details, specifieke limieten
- 2: Randvoorwaarden, kleine details
- 1: Administratief, definities, weinig klantwaarde

Antwoord ALLEEN in JSON: {"items": [...]}"""


FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "SECTIE: WA > Wat is verzekerd\n---\n"
            "[clause_id=DOC1:c001]\n"
            "Schade die je auto veroorzaakt aan anderen is verzekerd.\n\n"
            "[clause_id=DOC1:c002]\n"
            "De maximale vergoeding is \u20ac 6,15 miljoen per gebeurtenis.\n\n"
            "[clause_id=DOC1:c003]\n"
            "Niet verzekerd is schade aan je eigen auto.\n\n"
            "[clause_id=DOC1:c004]\n"
            "Je moet schade zo snel mogelijk aan ons melden.\n\n"
            "[clause_id=DOC1:c005]\n"
            "Bij inbraak of vandalisme moet je binnen 14 dagen aangifte doen bij de politie.\n---"
        ),
    },
    {
        "role": "assistant",
        "content": '{"items": ['
        '{"item_type": "covered", "module": "wa", '
        '"customer_facing_headline": "Schade aan anderen verzekerd", '
        '"exact_quote": "Schade die je auto veroorzaakt aan anderen is verzekerd.", '
        '"supporting_clause_ids": ["DOC1:c001"], '
        '"importance": 5, "theme": "liability", '
        '"exclusion_scope": [], '
        '"money_amounts": [], "deadlines": [], "conditions": []},'
        '{"item_type": "limit", "module": "wa", '
        '"customer_facing_headline": null, '
        '"exact_quote": "De maximale vergoeding is \\u20ac 6,15 miljoen per gebeurtenis.", '
        '"supporting_clause_ids": ["DOC1:c002"], '
        '"importance": 4, "theme": "liability", '
        '"exclusion_scope": [], '
        '"money_amounts": ["\\u20ac 6,15 miljoen"], "deadlines": [], "conditions": []},'
        '{"item_type": "not_covered", "module": "wa", '
        '"customer_facing_headline": "Schade aan eigen auto niet gedekt", '
        '"exact_quote": "Niet verzekerd is schade aan je eigen auto.", '
        '"supporting_clause_ids": ["DOC1:c003"], '
        '"importance": 4, "theme": "own_damage", '
        '"exclusion_scope": ["wa"], '
        '"money_amounts": [], "deadlines": [], "conditions": []},'
        '{"item_type": "notification_duty", "module": "general", '
        '"customer_facing_headline": "Schade zo snel mogelijk melden", '
        '"exact_quote": "Je moet schade zo snel mogelijk aan ons melden.", '
        '"supporting_clause_ids": ["DOC1:c004"], '
        '"importance": 3, "theme": "damage_reporting", '
        '"exclusion_scope": [], '
        '"money_amounts": [], "deadlines": [], "conditions": []},'
        '{"item_type": "notification_duty", "module": "general", '
        '"customer_facing_headline": "Aangifte bij politie binnen 14 dagen", '
        '"exact_quote": "Bij inbraak of vandalisme moet je binnen 14 dagen aangifte doen bij de politie.", '
        '"supporting_clause_ids": ["DOC1:c005"], '
        '"importance": 4, "theme": "police_reporting", '
        '"exclusion_scope": [], '
        '"money_amounts": [], "deadlines": ["14 dagen"], "conditions": []}'
        ']}',
    },
]


def build_extraction_messages(
    context_text: str,
    section_path: str,
    is_continuation: bool = False,
) -> list[dict[str, str]]:
    """Build the message list for a section extraction call."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_EXTRACTION},
    ]
    messages.extend(FEW_SHOT_EXAMPLES)

    continuation_note = ""
    if is_continuation:
        continuation_note = (
            "\n(Dit is een vervolg van de vorige sectie. "
            "Sommige clausules zijn al eerder geanalyseerd.)\n"
        )

    user_msg = f"SECTIE: {section_path}\n{continuation_note}---\n{context_text}\n---"
    messages.append({"role": "user", "content": user_msg})
    return messages
