from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib


@dataclass
class DeepModelSpec:
    encoder_name: str
    use_style: bool
    tuning: str
    style_dim: int = 12
    hidden_dim: int = 256
    dropout: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_targets: tuple[str, ...] = ("query", "value")


def require_deep_dependencies():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Install CUDA PyTorch, then run: python -m pip install -r requirements-deep.txt"
        ) from exc
    return torch, AutoModel, AutoTokenizer


def build_encoder(spec: DeepModelSpec):
    _, AutoModel, _ = require_deep_dependencies()
    encoder = AutoModel.from_pretrained(spec.encoder_name)
    if hasattr(encoder.config, "use_cache"):
        encoder.config.use_cache = False
    if spec.tuning == "lora":
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError("LoRA requires the peft package.") from exc
        encoder = get_peft_model(
            encoder,
            LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=spec.lora_r,
                lora_alpha=spec.lora_alpha,
                lora_dropout=spec.lora_dropout,
                target_modules=list(spec.lora_targets),
                bias="none",
            ),
        )
    elif spec.tuning != "full":
        raise ValueError("tuning must be 'full' or 'lora'")
    return encoder


def _hidden_size(encoder) -> int:
    config = getattr(encoder, "config", None)
    if config is not None and hasattr(config, "hidden_size"):
        return int(config.hidden_size)
    if hasattr(encoder, "get_base_model"):
        return int(encoder.get_base_model().config.hidden_size)
    raise AttributeError("Could not determine encoder hidden size.")


def create_model(spec: DeepModelSpec, encoder=None):
    torch, _, _ = require_deep_dependencies()
    from transformers.modeling_outputs import SequenceClassifierOutput

    class EncoderStyleClassifier(torch.nn.Module):
        def __init__(self, model_spec: DeepModelSpec, base_encoder):
            super().__init__()
            self.spec = model_spec
            self.encoder = base_encoder
            input_dim = _hidden_size(base_encoder) + (model_spec.style_dim if model_spec.use_style else 0)
            self.classifier = torch.nn.Sequential(
                torch.nn.LayerNorm(input_dim),
                torch.nn.Dropout(model_spec.dropout),
                torch.nn.Linear(input_dim, model_spec.hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout(model_spec.dropout),
                torch.nn.Linear(model_spec.hidden_dim, 2),
            )

        def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                kwargs = gradient_checkpointing_kwargs or {}
                self.encoder.gradient_checkpointing_enable(**kwargs)

        def gradient_checkpointing_disable(self):
            if hasattr(self.encoder, "gradient_checkpointing_disable"):
                self.encoder.gradient_checkpointing_disable()

        def forward(self, input_ids=None, attention_mask=None, style_features=None, labels=None, **kwargs):
            accepted = {key: value for key, value in kwargs.items() if key in {"token_type_ids"}}
            outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **accepted
            )
            pooled = outputs.last_hidden_state[:, 0]
            if self.spec.use_style:
                if style_features is None:
                    raise ValueError("style_features are required for fusion mode")
                pooled = torch.cat([pooled, style_features.to(pooled.dtype)], dim=-1)
            logits = self.classifier(pooled)
            loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return SequenceClassifierOutput(loss=loss, logits=logits)

    return EncoderStyleClassifier(spec, encoder if encoder is not None else build_encoder(spec))


def save_checkpoint(model, tokenizer, scaler, output_dir: str | Path) -> None:
    torch, _, _ = require_deep_dependencies()
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    # model.encoder.save_pretrained(target / "encoder")
    state_dict = {
    key: value.contiguous()
    for key, value in model.encoder.state_dict().items()
    }
    model.encoder.save_pretrained(
        target / "encoder",
        state_dict=state_dict,
    )
    tokenizer.save_pretrained(target / "tokenizer")
    torch.save(model.classifier.state_dict(), target / "classifier.pt")
    (target / "model_spec.json").write_text(
        json.dumps(asdict(model.spec), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    scaler_path = target / "style_scaler.joblib"
    if scaler is not None:
        joblib.dump(scaler, scaler_path)
    elif scaler_path.exists():
        scaler_path.unlink()


def load_checkpoint(checkpoint_dir: str | Path):
    torch, AutoModel, AutoTokenizer = require_deep_dependencies()
    target = Path(checkpoint_dir)
    raw = json.loads((target / "model_spec.json").read_text(encoding="utf-8"))
    raw["lora_targets"] = tuple(raw.get("lora_targets", ("query", "value")))
    spec = DeepModelSpec(**raw)
    if spec.tuning == "lora":
        from peft import PeftModel
        encoder = PeftModel.from_pretrained(
            AutoModel.from_pretrained(spec.encoder_name), target / "encoder"
        )
    else:
        encoder = AutoModel.from_pretrained(target / "encoder")
    model = create_model(spec, encoder=encoder)
    state = torch.load(target / "classifier.pt", map_location="cpu", weights_only=True)
    model.classifier.load_state_dict(state)
    tokenizer = AutoTokenizer.from_pretrained(target / "tokenizer")
    scaler_path = target / "style_scaler.joblib"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    return model, tokenizer, scaler, spec


def parameter_counts(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}
