from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT / "src" / "aigc_detector"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from deep_models import load_checkpoint, require_deep_dependencies  # noqa: E402
from evaluate import positive_scores  # noqa: E402
from explain import explain_linear_text  # noqa: E402
from features import StylometricTransformer  # noqa: E402
from predict import risk_band  # noqa: E402


st.set_page_config(page_title="AIGC 文本检测 Demo", page_icon="🔎", layout="wide")


@dataclass(frozen=True)
class ModelEntry:
    kind: str
    path: Path
    label: str
    result_path: Path | None


RISK_LABELS = {
    "low_ai_risk": "低 AI 风险",
    "uncertain_review_needed": "不确定，需人工复核",
    "high_ai_risk": "高 AI 风险",
}


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def classical_result_path(model_path: Path) -> Path | None:
    try:
        parts = model_path.relative_to(MODELS_DIR).parts
    except ValueError:
        return None
    if len(parts) == 3 and model_path.name.startswith("seed_"):
        return RESULTS_DIR / parts[0] / parts[1] / f"{model_path.stem}.json"
    return None


def deep_result_path(checkpoint_path: Path) -> Path | None:
    try:
        parts = checkpoint_path.relative_to(MODELS_DIR).parts
    except ValueError:
        return None
    if len(parts) == 4 and parts[0] == "deep":
        return RESULTS_DIR / parts[1] / parts[2] / f"{parts[3]}.json"
    return None


def entry_label(kind: str, path: Path, result_path: Path | None) -> str:
    result = read_json(result_path)
    benchmark = str(result.get("benchmark") or "unknown")
    method = str(result.get("method") or path.parent.name)
    seed = result.get("seed")
    seed_text = f"seed {seed}" if seed is not None else path.stem
    family = "深度" if kind == "deep" else "传统"
    return f"{family} · {benchmark} · {method} · {seed_text}"


def discover_models() -> list[ModelEntry]:
    entries: list[ModelEntry] = []

    for path in sorted(MODELS_DIR.glob("*/*/seed_*.joblib")):
        result_path = classical_result_path(path)
        entries.append(ModelEntry("classical", path, entry_label("classical", path, result_path), result_path))

    # Keep compatibility with manually exported top-level joblib models.
    for path in sorted(MODELS_DIR.glob("*.joblib")):
        entries.append(ModelEntry("classical", path, f"传统 · local · {path.stem}", None))

    for spec_path in sorted((MODELS_DIR / "deep").glob("*/*/seed_*/model_spec.json")):
        checkpoint = spec_path.parent
        required = [checkpoint / "classifier.pt", checkpoint / "encoder", checkpoint / "tokenizer"]
        if not all(path.exists() for path in required):
            continue
        result_path = deep_result_path(checkpoint)
        entries.append(ModelEntry("deep", checkpoint, entry_label("deep", checkpoint, result_path), result_path))

    return sorted(entries, key=lambda item: (item.kind != "deep", item.label, str(item.path)))


@st.cache_resource(show_spinner=False, max_entries=4)
def load_classical_model(path: str):
    # joblib can execute code while loading. Only load trusted local artifacts.
    return joblib.load(path)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_deep_model(path: str):
    torch, _, _ = require_deep_dependencies()
    model, tokenizer, scaler, spec = load_checkpoint(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, tokenizer, scaler, spec, device


def deep_probability(bundle, text: str, max_length: int) -> tuple[float, np.ndarray, np.ndarray | None]:
    model, tokenizer, scaler, spec, device = bundle
    torch, _, _ = require_deep_dependencies()
    encoded = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    extractor = StylometricTransformer()
    raw_style = extractor.transform([text])
    scaled_style = None
    if spec.use_style:
        if scaler is None:
            raise RuntimeError("Fusion checkpoint 缺少 style_scaler.joblib。")
        scaled_style = scaler.transform(raw_style).astype(np.float32)
        encoded["style_features"] = torch.tensor(scaled_style, dtype=torch.float32, device=device)

    with torch.inference_mode():
        logits = model(**encoded).logits
        probability = torch.softmax(logits, dim=-1)[0, 1].item()
    return float(probability), raw_style[0], None if scaled_style is None else scaled_style[0]


def metric_text(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def render_experiment_record(entry: ModelEntry, result: dict) -> None:
    with st.expander("模型与固定测试集记录", expanded=False):
        st.code(relative_path(entry.path), language=None)
        if not result:
            st.info("没有找到与该模型对应的 schema-v1 结果 JSON。")
            return

        overall = result.get("overall", {})
        columns = st.columns(4)
        columns[0].metric("Test Accuracy", metric_text(overall.get("accuracy")))
        columns[1].metric("Test F1", metric_text(overall.get("f1")))
        columns[2].metric("Test AUROC", metric_text(overall.get("auroc")))
        columns[3].metric("Human FPR", metric_text(overall.get("human_fpr")))

        details = {
            "benchmark": result.get("benchmark"),
            "method": result.get("method"),
            "seed": result.get("seed"),
            "encoder": result.get("encoder"),
            "tuning": result.get("tuning"),
            "fusion": result.get("fusion"),
            "classification_threshold": result.get("threshold", 0.5),
            "result_json": relative_path(entry.result_path) if entry.result_path else None,
        }
        st.json(details)
        st.caption("以上指标来自固定测试集，不代表当前输入文本的真实性结论。")


def render_style_features(raw_values: np.ndarray, scaled_values: np.ndarray | None) -> None:
    names = StylometricTransformer().get_feature_names_out()
    data = {"feature": names, "raw_value": raw_values}
    if scaled_values is not None:
        data["standardized_value"] = scaled_values
    frame = pd.DataFrame(data)
    st.subheader("风格特征")
    st.dataframe(frame, width="stretch", hide_index=True)
    st.caption("Fusion 模型使用标准化后的十二维风格特征；这些数值不是单独的因果解释。")


st.title("AIGC 文本检测 Demo")
st.caption("统一加载项目中的传统模型和深度 checkpoint；0=human，1=machine。")

entries = discover_models()
if not entries:
    st.error("没有发现可用模型。请先运行训练脚本生成 models/ 下的正式模型或 checkpoint。")
    st.code(
        "python scripts/run_experiments.py --profile course --benchmarks hc3 --families classical",
        language="powershell",
    )
    st.stop()

with st.sidebar:
    st.header("模型设置")
    entries_by_label = {entry.label: entry for entry in entries}
    selected_label = st.selectbox("选择模型", options=list(entries_by_label))
    selected = entries_by_label[selected_label]
    result = read_json(selected.result_path)
    st.caption(f"模型路径：`{relative_path(selected.path)}`")

    if selected.kind == "deep":
        spec_preview = read_json(selected.path / "model_spec.json")
        default_length = int(result.get("hyperparameters", {}).get("max_length", 256))
        max_length = st.number_input("最大 token 长度", min_value=32, max_value=1024, value=default_length, step=32)
        st.write(f"编码器：`{spec_preview.get('encoder_name', 'unknown')}`")
        st.write(f"微调：`{spec_preview.get('tuning', 'unknown')}`")
        st.write(f"融合风格特征：`{bool(spec_preview.get('use_style', False))}`")
    else:
        max_length = 256

    threshold = float(result.get("threshold", 0.5))
    st.write(f"分类阈值：`{threshold:.2f}`")

render_experiment_record(selected, result)

default_text = "本文围绕网络与信息安全中的 AIGC 内容鉴别问题展开分析，并讨论跨域泛化、改写扰动和人类误报风险。"
with st.form("prediction_form"):
    text = st.text_area("输入待检测文本", value=default_text, height=220)
    submitted = st.form_submit_button("检测", type="primary", width="stretch")

if submitted:
    clean_text = text.strip()
    if not clean_text:
        st.warning("请输入非空文本。")
    else:
        try:
            with st.spinner("正在加载模型并执行推理……"):
                if selected.kind == "classical":
                    model = load_classical_model(str(selected.path))
                    score = float(positive_scores(model, [clean_text])[0])
                    explanation = explain_linear_text(model, clean_text)
                    raw_style = scaled_style = None
                    device_text = "CPU"
                else:
                    bundle = load_deep_model(str(selected.path))
                    score, raw_style, scaled_style = deep_probability(bundle, clean_text, int(max_length))
                    explanation = None
                    device_text = str(bundle[-1]).upper()

            band = risk_band(score)
            predicted_label = int(score >= threshold)
            columns = st.columns(4)
            columns[0].metric("AI 分数", f"{score:.2%}")
            columns[1].metric("预测标签", "机器生成" if predicted_label else "人类撰写")
            columns[2].metric("风险区间", RISK_LABELS[band])
            columns[3].metric("推理设备", device_text)
            st.progress(min(max(score, 0.0), 1.0))

            if band == "uncertain_review_needed":
                st.warning("模型处于不确定区间，应进行人工复核。")
            elif band == "high_ai_risk":
                st.error("模型给出高 AI 风险分数，但不能据此自动定责。")
            else:
                st.success("模型给出低 AI 风险分数，仍应结合文本来源和写作过程判断。")

            if explanation is not None:
                st.subheader("线性模型特征贡献")
                if "top_features" in explanation:
                    st.dataframe(pd.DataFrame(explanation["top_features"]), width="stretch", hide_index=True)
                else:
                    st.info(explanation.get("error", "当前传统模型不支持特征贡献展示。"))
            elif raw_style is not None and scaled_style is not None:
                render_style_features(raw_style, scaled_style)
            elif selected.kind == "deep":
                st.info("Encoder-only 模型当前只输出分类分数，项目尚未实现 token 级归因解释。")

        except Exception as exc:  # pragma: no cover - Streamlit runtime diagnostics
            st.error(f"模型加载或推理失败：{exc}")
            with st.expander("错误详情"):
                st.exception(exc)

st.divider()
st.subheader("使用边界")
st.write(
    "AI 分数来自当前模型和训练分布，尚未经过面向实际应用的概率校准。AIGC 检测不应作为处分、定责、"
    "学术不端认定或内容下架的唯一依据；请结合写作过程、版本历史、文本来源和人工复核。"
)
