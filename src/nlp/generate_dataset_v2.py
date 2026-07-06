"""
SkinScan AI — NLP Symptoms Module (Member 4)
generate_dataset_v2.py

REBUILT for the actual 10 CV classes (per class_distribution.png), English-only
(module is going 100% BioBERT, which has no Arabic vocabulary — this
deliberately drops Arabic support; flag this decision to the team since it
changes how FR-11 is satisfied, or isn't, for this module).

Class name strings are copied verbatim from the CV team's chart (including
parenthetical abbreviations) so they match Member 2's classifier output
exactly — this was flagged repeatedly as the #1 integration risk.
"""

import random
import csv

random.seed(42)

CONDITIONS = {
    "Eczema": {
        "symptom_type": ["dry scaly patches", "redness", "weeping skin", "cracked skin", "flaking"],
        "triggers": ["allergens", "dry weather", "certain fabrics", "soaps", "stress"],
        "locations": ["inner elbow", "behind the knees", "neck", "hands", "face"],
        "duration_range": (7, 120),
        "spreading_prob": 0.45,
        "pain_levels": ["none", "mild"],
        "itch_levels": ["moderate", "severe"],
        "base_severity": 0.30,
    },
    "Melanoma": {
        "symptom_type": ["changing mole", "new dark lesion", "bleeding spot", "irregular border",
                          "asymmetric mole", "uneven mole color", "mole growing larger"],
        "triggers": ["sun exposure", "history of sunburn", "family history of melanoma", "fair skin"],
        "locations": ["back", "arm", "leg", "shoulder", "face", "chest"],
        "duration_range": (14, 180),
        "spreading_prob": 0.35,
        "pain_levels": ["none", "mild"],
        "itch_levels": ["none", "mild"],
        "base_severity": 0.85,
    },
    "Atopic Dermatitis": {
        "symptom_type": ["chronic itchy skin", "thickened skin from scratching", "dry inflamed patches",
                          "redness", "small fluid-filled bumps"],
        "triggers": ["allergens", "stress", "sweating", "wool clothing", "weather changes", "dust mites"],
        "locations": ["inner elbows", "behind the knees", "wrists", "ankles", "face", "neck"],
        "duration_range": (30, 365),
        "spreading_prob": 0.4,
        "pain_levels": ["none", "mild"],
        "itch_levels": ["severe"],
        "base_severity": 0.4,
    },
    "Basal Cell Carcinoma (BCC)": {
        "symptom_type": ["pearly bump", "waxy translucent nodule", "visible small blood vessels",
                          "flat scar-like patch", "sore that won't heal", "sore that bleeds and returns"],
        "triggers": ["long-term sun exposure", "fair skin", "older age", "prior radiation exposure"],
        "locations": ["face", "ears", "neck", "scalp", "shoulders", "back"],
        "duration_range": (60, 365),
        "spreading_prob": 0.25,
        "pain_levels": ["none", "mild"],
        "itch_levels": ["none", "mild"],
        "base_severity": 0.6,
    },
    "Melanocytic Nevi (NV)": {
        "symptom_type": ["small round mole", "symmetric brown spot", "flat stable mole",
                          "uniform colored mole", "slightly raised mole"],
        "triggers": ["none noticed", "sun exposure over the years", "genetics"],
        "locations": ["arm", "back", "chest", "leg", "face", "shoulder"],
        "duration_range": (180, 2000),
        "spreading_prob": 0.05,
        "pain_levels": ["none"],
        "itch_levels": ["none", "mild"],
        "base_severity": 0.1,
    },
    "Benign Keratosis-like Lesions (BKL)": {
        "symptom_type": ["waxy stuck-on patch", "rough warty texture", "tan or brown plaque",
                          "well-defined scaly patch", "dull scaly surface"],
        "triggers": ["aging", "sun exposure", "genetics"],
        "locations": ["chest", "back", "face", "scalp", "shoulder"],
        "duration_range": (90, 1000),
        "spreading_prob": 0.15,
        "pain_levels": ["none"],
        "itch_levels": ["none", "mild"],
        "base_severity": 0.25,
    },
    "Psoriasis pictures Lichen Planus and related diseases": {
        "symptom_type": ["silvery plaques", "thick red patches", "scalp flaking", "joint pain",
                          "itchy purple flat-topped bumps", "polygonal itchy papules", "cracked dry skin"],
        "triggers": ["stress", "family history", "cold weather", "skin injury", "infection", "certain medications"],
        "locations": ["knees", "elbows", "scalp", "lower back", "wrists", "ankles", "nails"],
        "duration_range": (14, 365),
        "spreading_prob": 0.45,
        "pain_levels": ["mild", "moderate"],
        "itch_levels": ["moderate", "severe"],
        "base_severity": 0.45,
    },
    "Seborrheic Keratoses and other Benign Tumors": {
        "symptom_type": ["waxy stuck-on growth", "warty rough bump", "brown or black growth",
                          "round scaly growth", "multiple scattered growths"],
        "triggers": ["aging", "genetics", "sun exposure"],
        "locations": ["chest", "back", "scalp", "face", "shoulder"],
        "duration_range": (180, 2000),
        "spreading_prob": 0.1,
        "pain_levels": ["none"],
        "itch_levels": ["none", "mild"],
        "base_severity": 0.2,
    },
    "Tinea Ringworm Candidiasis and other Fungal Infections": {
        "symptom_type": ["circular scaly ring", "raised red border", "clear center", "moist red rash",
                          "satellite red bumps", "soreness in skin folds"],
        "triggers": ["contact with infected person", "contact with animal", "shared towels",
                     "humid environment", "sweating", "tight clothing"],
        "locations": ["arm", "leg", "torso", "scalp", "foot", "underarm", "groin fold"],
        "duration_range": (3, 45),
        "spreading_prob": 0.55,
        "pain_levels": ["none", "mild"],
        "itch_levels": ["moderate", "severe"],
        "base_severity": 0.2,
    },
    "Warts Molluscum and other Viral Infections": {
        "symptom_type": ["rough cauliflower-like bump", "small dome-shaped bump", "pearly bump with a central dimple",
                          "hard raised bump", "cluster of small bumps"],
        "triggers": ["skin-to-skin contact", "shared personal items", "weakened immune system",
                     "public swimming pools", "scratching or picking"],
        "locations": ["hands", "fingers", "feet", "face", "trunk", "neck"],
        "duration_range": (14, 180),
        "spreading_prob": 0.5,
        "pain_levels": ["none", "mild"],
        "itch_levels": ["none", "mild"],
        "base_severity": 0.15,
    },
}

EN_TEMPLATES = [
    "I have {symptoms} on my {location} for about {duration} days, {spread_clause} triggered by {trigger}.",
    "There is {symptoms} on my {location}. It started {duration} days ago and {spread_clause}",
    "My {location} shows {symptoms}, duration around {duration} days. It seems to be {trigger} related.",
    "{symptoms} appeared on my {location} roughly {duration} days ago, {spread_clause}",
    "I've noticed {symptoms} near my {location} for {duration} days now, worse after {trigger}.",
    "Experiencing {symptoms} on the {location} area, been {duration} days, {spread_clause}",
]

SPREAD_CLAUSES_EN = ["and it seems to be spreading.", "it hasn't spread so far.", "and it's spreading slowly."]


def severity_label(score: float) -> str:
    if score >= 0.8:
        return "urgent"
    elif score >= 0.55:
        return "severe"
    elif score >= 0.35:
        return "moderate"
    else:
        return "mild"


def compute_severity(base: float, duration: int, spreading: bool, pain: str, itch: str) -> float:
    score = base
    if duration > 45:
        score += 0.08
    if spreading:
        score += 0.12
    if pain in ("moderate", "severe"):
        score += 0.08
    if itch == "severe":
        score += 0.04
    score += random.uniform(-0.05, 0.05)
    return round(max(0.0, min(score, 1.0)), 2)


def make_row(condition: str):
    cfg = CONDITIONS[condition]
    n_symptoms = random.randint(1, 3)
    symptom_pool = cfg["symptom_type"]
    trigger_pool = cfg["triggers"]
    location_pool = cfg["locations"]

    chosen_symptoms = random.sample(symptom_pool, min(n_symptoms, len(symptom_pool)))
    duration = random.randint(*cfg["duration_range"])
    spreading = random.random() < cfg["spreading_prob"]
    pain = random.choice(cfg["pain_levels"])
    itch = random.choice(cfg["itch_levels"])
    trigger = random.choice(trigger_pool)
    location = random.choice(location_pool)
    spread_clause = random.choice(SPREAD_CLAUSES_EN) if spreading else "it hasn't spread so far."

    template = random.choice(EN_TEMPLATES)
    text = template.format(
        symptoms=", ".join(chosen_symptoms),
        location=location,
        duration=duration,
        trigger=trigger,
        spread_clause=spread_clause,
    )

    score = compute_severity(cfg["base_severity"], duration, spreading, pain, itch)
    label = severity_label(score)
    canonical_tags = cfg["symptom_type"][:len(chosen_symptoms)]

    return {
        "text": text,
        "language": "en",
        "condition": condition,
        "symptom_type": "|".join(canonical_tags),
        "duration_days": duration,
        "spreading": spreading,
        "pain_level": pain,
        "itch_level": itch,
        "triggers": trigger,
        "severity_score": score,
        "severity_label": label,
    }


def generate_dataset(rows_per_condition=200):
    rows = []
    for condition in CONDITIONS:
        for _ in range(rows_per_condition):
            rows.append(make_row(condition))
    random.shuffle(rows)
    return rows


if __name__ == "__main__":
    rows = generate_dataset(rows_per_condition=200)
    fieldnames = [
        "text", "language", "condition", "symptom_type", "duration_days",
        "spreading", "pain_level", "itch_level", "triggers",
        "severity_score", "severity_label",
    ]
    out_path = "/home/claude/nlp_symptoms/data/symptom_dataset_v2.csv"
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Generated {len(rows)} rows -> {out_path}")
    print(f"Conditions: {list(CONDITIONS.keys())}")
    print(f"Rows per condition: {len(rows)//len(CONDITIONS)}")
