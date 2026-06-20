from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src" / "aigc_detector"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deep_models import load_checkpoint, require_deep_dependencies
from features import StylometricTransformer


def parse_args():
    parser = argparse.ArgumentParser(description="Predict with a deep AIGC detector checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    torch, _, _ = require_deep_dependencies()
    model, tokenizer, scaler, spec = load_checkpoint(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    encoded = tokenizer(args.text, truncation=True, max_length=args.max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    style = StylometricTransformer().transform([args.text])
    if scaler is not None:
        style = scaler.transform(style)
    encoded["style_features"] = torch.tensor(style, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(**encoded).logits
        probability = torch.softmax(logits, dim=-1)[0, 1].item()
    risk = "high_ai_risk" if probability >= 0.75 else "low_ai_risk" if probability <= 0.25 else "uncertain_review_needed"
    print(f"ai_probability={probability:.4f}")
    print(f"predicted_label={int(probability >= 0.5)}")
    print(f"risk_band={risk}")
    print(f"encoder={spec.encoder_name}")
    print(f"fusion={spec.use_style}")


if __name__ == "__main__":
    main()
