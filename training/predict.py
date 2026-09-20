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

FIELD_ORDER = list(FIELD_HELP.keys())
def validate_field(field, value):
    choices = {
        "RIAGENDR": {1, 2},
        "RIDRETH1": {1, 2, 3, 4, 5},
        "DMDEDUC2": {1, 2, 3, 4, 5},
        "DMDMARTL": {1, 2, 3, 4, 5, 6},
        "DMDCITZN": {1, 2},
    }

    ranges = {
        "RIDAGEYR": (18, 85),
        "INDFMPIR": (0, 5),
        "DMDHHSIZ": (1, 7),
    }

    if field in choices and value not in choices[field]:
        raise ValueError(f"Invalid value for {field}: {value}")

    if field in ranges and not ranges[field][0] <= value <= ranges[field][1]:
        raise ValueError(f"Invalid value for {field}: {value}")

    return value


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Predict survival percentage for a custom person using the best trained model."
    )

    for field, help_text in FIELD_HELP.items():
        parser.add_argument(f"--{field.lower()}", dest=field, 
                            type=lambda value, field=field: validate_field(field,
                             float(value)), help=help_text)

    return parser


def prompt_for_missing(person):
    for field in FIELD_ORDER:
        if person.get(field) is not None:
            continue

        while True:
            raw_value = input(f"{field} ({FIELD_HELP[field]}): ").strip()
            try:
                person[field] = validate_field(field, float(raw_value))
                break
            except ValueError:
                print("Please enter a number.")

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
