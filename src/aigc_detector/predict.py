from __future__ import annotations

import argparse

import joblib

from evaluate import positive_scores
from explain import explain_linear_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict whether a text is AI-generated.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--explain", action="store_true")
    return parser.parse_args()


def risk_band(score: float) -> str:
    if score >= 0.75:
        return "high_ai_risk"
    if score <= 0.25:
        return "low_ai_risk"
    return "uncertain_review_needed"


def main() -> None:
    args = parse_args()
    model = joblib.load(args.model)
    score = float(positive_scores(model, [args.text])[0])
    label = int(score >= 0.5)
    print(f"ai_probability={score:.4f}")
    print(f"predicted_label={label}")
    print(f"risk_band={risk_band(score)}")
    if args.explain:
        print(explain_linear_text(model, args.text))


if __name__ == "__main__":
    main()
