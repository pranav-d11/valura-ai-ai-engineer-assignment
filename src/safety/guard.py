import re

from src.models import SafetyCategory, SafetyResult


EDUCATIONAL_MARKERS = [
    "how does",
    "what is",
    "explain",
    "define",
    "teach me",
    "example of",
    "history of",
    "penalty for",
    "why is",
    "difference between",
]


BLOCK_RULES: dict[SafetyCategory, tuple[list[str], str]] = {
    SafetyCategory.INSIDER_TRADING: (
        [
            "material non-public",
            "inside information",
            "tip about earnings",
            "confidential merger news",
            "before tomorrow's announcement",
            "unannounced acquisition",
            "friend at microsoft gave me a tip",
            "work at apple and know",
            "load up before",
        ],
        "This request involves acting on material non-public information, which constitutes insider trading under securities law. Valura cannot assist with this.",
    ),
    SafetyCategory.MARKET_MANIPULATION: (
        [
            "pump and dump",
            "pump up the price",
            "coordinated buying scheme",
            "wash trade",
            "wash trading between two accounts",
            "create volume",
            "move this stock 20%",
        ],
        "This request describes market manipulation, which is illegal under securities regulations. Valura cannot assist with this.",
    ),
    SafetyCategory.MONEY_LAUNDERING: (
        [
            "avoid the 10k reporting threshold",
            "obscure the source of these funds",
            "hide trading profits from the tax authorities",
            "route trades through a shell company",
            "without reporting it",
            "launder",
            "structure deposits",
        ],
        "This request appears to involve concealing or laundering funds, which is a serious financial crime. Valura cannot assist with this.",
    ),
    SafetyCategory.GUARANTEED_RETURNS: (
        [
            "guarantee me",
            "promise me my money will double",
            "100% certain to go up",
            "foolproof way to make",
            "risk-free profit",
            "can't lose",
            "cannot lose",
        ],
        "No investment can guarantee returns. Valura does not endorse or facilitate any claims of guaranteed or risk-free profits.",
    ),
    SafetyCategory.RECKLESS_ADVICE: (
        [
            "put all my retirement savings in crypto",
            "take a margin loan",
            "entire emergency fund into options",
            "mortgage my house",
            "bet everything",
            "leverage everything",
        ],
        "This request involves a level of financial risk that falls outside what Valura can responsibly recommend. Please consult a certified financial advisor.",
    ),
    SafetyCategory.SANCTIONS_EVASION: (
        [
            "bypass ofac sanctions",
            "sanctioned russian company",
            "without it being traced",
            "evade sanctions",
        ],
        "This request appears to involve sanctions evasion, which may violate financial crime and sanctions laws. Valura cannot assist with this.",
    ),
    SafetyCategory.FRAUD: (
        ["draft a fake contract note", "forge", "fake statement"],
        "This request appears to involve fraud or document falsification. Valura cannot assist with this.",
    ),
}


def run_safety_guard(query: str) -> SafetyResult:
    if not query:
        return SafetyResult(blocked=False, category=SafetyCategory.CLEAN)

    lowered = query.lower().strip()
    is_educational = any(marker in lowered for marker in EDUCATIONAL_MARKERS)

    for category, (keywords, refusal) in BLOCK_RULES.items():
        for keyword in keywords:
            if re.search(re.escape(keyword), lowered):
                if is_educational:
                    return SafetyResult(blocked=False, category=SafetyCategory.CLEAN)
                return SafetyResult(blocked=True, category=category, message=refusal)

    return SafetyResult(blocked=False, category=SafetyCategory.CLEAN)


def check(query: str) -> SafetyResult:
    return run_safety_guard(query)
