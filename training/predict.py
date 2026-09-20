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

ALLOWED_CODES = {
    "RIAGENDR": [1, 2],
    "RIDRETH1": [1, 2, 3, 4, 5],
    "DMDEDUC2": [1, 2, 3, 4, 5],
    "DMDMARTL": [1, 2, 3, 4, 5, 6],
    "DMDCITZN": [1, 2],
}

ALLOWED_RANGE = {
    "RIDAGEYR": (18, 85),
    "INDFMPIR": (0, 5),
    "DMDHHSIZ": (1, 7),
}


def check_value(field, value):
    if field in ALLOWED_CODES and value not in ALLOWED_CODES[field]:
        codes = ", ".join(str(code) for code in ALLOWED_CODES[field])
        raise ValueError(f"{field} has to be one of {codes}")

    if field in ALLOWED_RANGE:
        low, high = ALLOWED_RANGE[field]
        if not low <= value <= high:
            raise ValueError(f"{field} has to be between {low} and {high}")

    return value


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Predict survival percentage for a custom person using the best trained model."
    )

    for field, help_text in FIELD_HELP.items():
        parser.add_argument(f"--{field.lower()}", dest=field, type=float, help=help_text)

    return parser


def prompt_for_missing(person):
    for field in FIELD_ORDER:
        if person.get(field) is not None:
            continue

        while True:
            raw_value = input(f"{field} ({FIELD_HELP[field]}): ").strip()
            try:
                value = float(raw_value)
            except ValueError:
                print("Please enter a number.")
                continue

            try:
                person[field] = check_value(field, value)
                break
            except ValueError as error:
                print(error)

    return person


def load_best_model():
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model found at {BEST_MODEL_PATH}. Run `py training/run_all.py` first."
        )
    return joblib.load(BEST_MODEL_PATH)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    person = {field: getattr(args, field) for field in FIELD_ORDER}

    for field, value in person.items():
        if value is not None:
            try:
                check_value(field, value)
            except ValueError as error:
                parser.error(str(error))

    person = prompt_for_missing(person)

    pipeline = load_best_model()
    survival_percentage = predict_survival_percentage(pipeline, person)

    print(f"\nEstimated survival probability: {survival_percentage:.1f}%")


if __name__ == "__main__":
    main()
