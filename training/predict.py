import argparse

import joblib

from run_all import BEST_MODEL_PATH, predict_survival_percentage


FIELD_HELP = {
    "RIDAGEYR": "Age in years",
    "RIAGENDR": "Gender: 1 = Male, 2 = Female",
    "RIDRETH1": (
        "Race/ethnicity: 1 = Mexican American, 2 = Other Hispanic, "
        "3 = Non-Hispanic White, 4 = Non-Hispanic Black, 5 = Other/Multi-racial"
    ),
    "DMDEDUC2": (
        "Education: 1 = Less than 9th grade, 2 = 9-11th grade, "
        "3 = High school grad/GED, 4 = Some college/AA degree, "
        "5 = College graduate or above"
    ),
    "DMDMARTL": (
        "Marital status: 1 = Married, 2 = Widowed, 3 = Divorced, "
        "4 = Separated, 5 = Never married, 6 = Living with partner"
    ),
    "INDFMPIR": "Family income-to-poverty ratio, 0 to 5 (5 = at or above 5x the poverty line)",
    "DMDCITZN": "Citizenship: 1 = Citizen, 2 = Not a citizen",
    "DMDHHSIZ": "People in household, 1 to 7 (7 = 7 or more)",
}

# Valid ranges for each field - matches the API validation in app/main.py
FIELD_RANGES = {
    "RIDAGEYR": (18, 85),
    "RIAGENDR": (1, 2),
    "RIDRETH1": (1, 5),
    "DMDEDUC2": (1, 5),
    "DMDMARTL": (1, 6),
    "INDFMPIR": (0, 5),
    "DMDCITZN": (1, 2),
    "DMDHHSIZ": (1, 7),
}

FIELD_ORDER = list(FIELD_HELP.keys())


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Predict survival percentage for a custom person using the best trained model."
    )

    for field, help_text in FIELD_HELP.items():
        parser.add_argument(f"--{field.lower()}", dest=field, type=float, help=help_text)

    return parser


def validate_field(field, value):
    """Validate that a value is within the acceptable range for the field."""
    if value is None:
        return False
    
    min_val, max_val = FIELD_RANGES[field]
    if not (min_val <= value <= max_val):
        print(f"Invalid value for {field}. Must be between {min_val} and {max_val}.")
        return False
    
    # For categorical fields, also check that it's an integer
    if field not in ["RIDAGEYR", "INDFMPIR"]:
        if not float(value).is_integer():
            print(f"Invalid value for {field}. Must be an integer.")
            return False
    
    return True

def prompt_for_missing(person):
    for field in FIELD_ORDER:
        # First validate any existing value from command line
        if person.get(field) is not None:
            if not validate_field(field, person[field]):
                person[field] = None  # Force re-prompt
        
        # Prompt for missing or invalid values
        if person.get(field) is None:
            while True:
                raw_value = input(f"{field} ({FIELD_HELP[field]}): ").strip()
                try:
                    value = float(raw_value)
                    if validate_field(field, value):
                        person[field] = value
                        break
                except ValueError:
                    print("Please enter a valid number.")

    return person


def load_best_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}. Run `py training/run_all.py` first."
        )
    return joblib.load(BEST_MODEL_PATH)


def main():
    args = build_arg_parser().parse_args()
    person = {field: getattr(args, field) for field in FIELD_ORDER}
    person = prompt_for_missing(person)

    pipeline = load_best_model()
    survival_percentage = predict_survival_percentage(pipeline, person)

    print(f"\nEstimated survival probability: {survival_percentage:.1f}%")


if __name__ == "__main__":
    main()
