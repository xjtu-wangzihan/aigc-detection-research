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

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #e8edf3;}
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e5eaf0;
        border-radius: 14px;
        padding: 0.85rem 1rem;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.045);
    }
    .hero {
        padding: 1.45rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(120deg, #eff6ff 0%, #f8fafc 52%, #ecfeff 100%);
        border: 1px solid #dbeafe;
        margin-bottom: 1.2rem;
    }
    .hero h1 {font-size: 2rem; margin: 0 0 0.35rem 0; color: #0f172a;}
    .hero p {margin: 0; color: #475569; line-height: 1.65;}
    .model-strip {
        display: flex; gap: 0.55rem; flex-wrap: wrap; align-items: center;
        margin: 0.25rem 0 1.1rem 0;
    }
    .pill {
        display: inline-block; padding: 0.25rem 0.65rem; border-radius: 999px;
        background: #f1f5f9; color: #334155; border: 1px solid #e2e8f0;
        font-size: 0.82rem;
    }
    .score-shell {margin: 0.7rem 0 0.25rem 0;}
    .score-track {
        position: relative; height: 18px; border-radius: 999px;
        background: linear-gradient(90deg, #22c55e 0 25%, #f59e0b 25% 75%, #ef4444 75% 100%);
        box-shadow: inset 0 0 0 1px rgba(15, 23, 42, 0.12);
    }
    .score-marker {
        position: absolute; top: -6px; width: 4px; height: 30px; border-radius: 4px;
        background: #0f172a; transform: translateX(-2px);
        box-shadow: 0 0 0 3px white, 0 2px 6px rgba(15, 23, 42, 0.3);
    }
    .score-labels {display: flex; justify-content: space-between; color: #64748b; font-size: 0.78rem; margin-top: 0.35rem;}
    .small-note {color: #64748b; font-size: 0.86rem; line-height: 1.55;}
    div[data-testid="stFormSubmitButton"] button {height: 2.9rem; border-radius: 10px; font-weight: 650;}
    </style>
    """,
    unsafe_allow_html=True,
)


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

BENCHMARK_LABELS = {
    "hc3": "HC3-English",
    "hc3_zh": "HC3-Chinese",
    "mage": "MAGE",
    "raid": "RAID",
    "local": "本地模型",
    "unknown": "未知数据集",
}

METHOD_LABELS = {
    "word_tfidf": "Word TF-IDF",
    "char_tfidf": "Char TF-IDF",
    "style": "Style-LR",
    "hybrid": "Hybrid-LR",
    "roberta_encoder": "RoBERTa Encoder",
    "roberta_style": "RoBERTa + Style",
    "xlmr_encoder": "XLM-R Encoder",
    "xlmr_style": "XLM-R + Style",
}

SAMPLE_TEXTS = {
    "学术说明": "本文围绕网络与信息安全中的生成文本鉴别问题展开分析，重点讨论跨领域泛化、文本扰动鲁棒性以及人类文本误报风险。实验采用统一的数据划分与评价指标，对多种轻量模型和预训练编码器进行比较。",
    "日常表达": "今天下班路上突然下雨了，我没带伞，只好在便利店门口等了一会儿。后来雨小了，路边的树叶特别亮，空气也凉快了不少。",
    "新闻风格": "研究团队发布了最新测试结果。报告显示，不同检测方法在多领域文本上的表现存在明显差异，研究人员建议在实际应用中保留人工复核环节。",
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
    family = "深度" if kind == "deep" else "传统"
    return f"{family} · {benchmark} · {method}"


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


def entry_metadata(entry: ModelEntry) -> tuple[str, str]:
    result = read_json(entry.result_path)
    benchmark = str(result.get("benchmark") or "local")
    method = str(result.get("method") or entry.path.parent.name or entry.path.stem)
    return benchmark, method


def model_option_label(entry: ModelEntry) -> str:
    _, method = entry_metadata(entry)
    return METHOD_LABELS.get(method, method)


def render_score_gauge(score: float) -> None:
    position = min(max(float(score), 0.0), 1.0) * 100
    st.markdown(
        f"""
        <div class="score-shell">
          <div class="score-track"><div class="score-marker" style="left:{position:.2f}%"></div></div>
          <div class="score-labels"><span>低风险 0–25%</span><span>需复核 25–75%</span><span>高风险 75–100%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_text_profile(text: str) -> None:
    compact = text.strip()
    characters = len(compact)
    lines = len([line for line in compact.splitlines() if line.strip()])
    whitespace_tokens = len(compact.split())
    chinese_chars = sum("\u4e00" <= char <= "\u9fff" for char in compact)
    st.caption(
        f"输入概况：{characters} 个字符 · {lines} 个非空行 · "
        f"{whitespace_tokens} 个空格分词 · {chinese_chars} 个汉字"
    )
    if characters < 50:
        st.warning("文本少于 50 个字符，短文本的风格与词汇证据不足，结果波动可能更大。")


def render_model_strip(entry: ModelEntry, result: dict) -> None:
    benchmark, method = entry_metadata(entry)
    family = "深度模型" if entry.kind == "deep" else "传统模型"
    chips = [
        family,
        BENCHMARK_LABELS.get(benchmark, benchmark),
        METHOD_LABELS.get(method, method),
    ]
    st.markdown(
        '<div class="model-strip">' + "".join(f'<span class="pill">{item}</span>' for item in chips) + "</div>",
        unsafe_allow_html=True,
    )

    overall = result.get("overall", {})
    if overall:
        columns = st.columns(4)
        columns[0].metric("固定测试集 F1", metric_text(overall.get("f1")))
        columns[1].metric("固定测试集 AUROC", metric_text(overall.get("auroc")))
        columns[2].metric("Human FPR", metric_text(overall.get("human_fpr")))
        columns[3].metric("测试样本数", str(overall.get("n", "—")))


def use_sample(name: str) -> None:
    st.session_state["analysis_text"] = SAMPLE_TEXTS[name]


def render_experiment_record(entry: ModelEntry, result: dict) -> None:
    with st.expander("模型与固定测试集记录", expanded=False):
        st.caption("当前记录来自所选本地模型及其配套结果文件。")
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
            "encoder": result.get("encoder"),
            "tuning": result.get("tuning"),
            "fusion": result.get("fusion"),
            "classification_threshold": result.get("threshold", 0.5),
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
    st.dataframe(frame, use_container_width=True, hide_index=True)
    st.caption("Fusion 模型使用标准化后的十二维风格特征；这些数值不是单独的因果解释。")


st.markdown(
    """
    <div class="hero">
      <h1>🔎 AIGC 文本检测实验台</h1>
      <p>比较传统特征模型与预训练编码器，对输入文本给出 AI 风险分数、复核区间和可用的特征解释。结果用于辅助研判，不用于自动定责。</p>
    </div>
    """,
    unsafe_allow_html=True,
)

entries = discover_models()
if not entries:
    st.error("没有发现可用模型。请先运行训练脚本生成 models/ 下的正式模型或 checkpoint。")
    st.code(
        "python scripts/run_experiments.py --profile course --benchmarks hc3 --families classical",
        language="powershell",
    )
    st.stop()

with st.sidebar:
    st.header("模型与推理设置")
    family_label = st.radio("模型类型", ["全部", "传统模型", "深度模型"], horizontal=True)
    family_kind = {"传统模型": "classical", "深度模型": "deep"}.get(family_label)
    family_entries = [entry for entry in entries if family_kind is None or entry.kind == family_kind]

    available_benchmarks = sorted(
        {entry_metadata(entry)[0] for entry in family_entries},
        key=lambda name: list(BENCHMARK_LABELS).index(name) if name in BENCHMARK_LABELS else 99,
    )
    benchmark_options = ["all", *available_benchmarks]
    benchmark = st.selectbox(
        "Benchmark",
        benchmark_options,
        format_func=lambda value: "全部 Benchmark" if value == "all" else BENCHMARK_LABELS.get(value, value),
    )
    filtered_entries = [
        entry for entry in family_entries if benchmark == "all" or entry_metadata(entry)[0] == benchmark
    ]
    selected = st.selectbox("选择模型", options=filtered_entries, format_func=model_option_label)
    result = read_json(selected.result_path)
    st.caption(f"已发现 {len(filtered_entries)} 个可选模型")

    if selected.kind == "deep":
        spec_preview = read_json(selected.path / "model_spec.json")
        default_length = int(result.get("hyperparameters", {}).get("max_length", 256))
        max_length = st.number_input(
            "最大 token 长度", min_value=32, max_value=1024, value=default_length, step=32,
            help="超过该长度的 token 会被截断。",
        )
    else:
        max_length = 256

    threshold = float(result.get("threshold", 0.5))
    with st.expander("当前模型配置", expanded=False):
        st.caption("模型文件已从本地 models/ 目录加载。")
        st.write(f"分类阈值：`{threshold:.2f}`")
        if selected.kind == "deep":
            st.write(f"编码器：`{spec_preview.get('encoder_name', 'unknown')}`")
            st.write(f"微调方式：`{spec_preview.get('tuning', 'unknown')}`")
            st.write(f"融合风格特征：`{bool(spec_preview.get('use_style', False))}`")
        overall_preview = result.get("overall", {})
        if overall_preview and float(overall_preview.get("human_fpr", 0) or 0) >= 0.1:
            st.warning("该模型固定测试集 Human FPR 较高，需特别关注人类文本误报。")

render_model_strip(selected, result)

st.subheader("输入待检测文本")
st.markdown('<div class="small-note">建议输入完整段落。短文本、模板文本、非母语写作和强改写文本的不确定性通常更高。</div>', unsafe_allow_html=True)
if "analysis_text" not in st.session_state:
    st.session_state["analysis_text"] = SAMPLE_TEXTS["学术说明"]

sample_columns = st.columns([1, 1, 1, 3])
for column, name in zip(sample_columns[:3], SAMPLE_TEXTS):
    column.button(name, on_click=use_sample, args=(name,), use_container_width=True)

with st.form("prediction_form"):
    text = st.text_area(
        "文本内容",
        key="analysis_text",
        height=230,
        label_visibility="collapsed",
        placeholder="粘贴或输入需要检测的完整文本……",
    )
    submitted = st.form_submit_button("开始检测", type="primary", use_container_width=True)

if submitted:
    clean_text = text.strip()
    if not clean_text:
        st.warning("请输入非空文本。")
    else:
        render_text_profile(clean_text)
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
            st.subheader("检测结果")
            columns = st.columns(4)
            columns[0].metric("AI 分数", f"{score:.2%}")
            columns[1].metric("预测标签", "机器生成" if predicted_label else "人类撰写")
            columns[2].metric("风险区间", RISK_LABELS[band])
            columns[3].metric("推理设备", device_text)
            render_score_gauge(score)

            overview_tab, explanation_tab, record_tab = st.tabs(["结果解读", "解释与特征", "模型记录"])
            with overview_tab:
                if band == "uncertain_review_needed":
                    st.warning("分数位于 25%～75% 的不确定区间，建议结合写作过程、版本记录和文本来源人工复核。")
                elif band == "high_ai_risk":
                    st.error("模型给出高 AI 风险分数，但该结果不是文本来源的确定性证据，不能据此自动定责。")
                else:
                    st.success("模型给出低 AI 风险分数，仍不能排除人工润色、混合写作或分布外文本。")
                st.write(
                    f"当前模型以 `{threshold:.2f}` 为分类阈值，AI 分数为 `{score:.4f}`，"
                    f"因此输出“{'机器生成' if predicted_label else '人类撰写'}”标签。"
                )
                st.caption("风险区间使用固定的 0.25 / 0.75 边界，与模型二分类阈值用途不同。")

            with explanation_tab:
                if explanation is not None and "top_features" in explanation:
                    st.markdown("#### 线性模型特征贡献")
                    feature_frame = pd.DataFrame(explanation["top_features"])
                    feature_frame["direction"] = feature_frame["direction"].map(
                        {"AI": "推向 AI", "Human": "推向人类"}
                    )
                    st.dataframe(
                        feature_frame,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "feature": "特征",
                            "value": st.column_config.NumberColumn("特征值", format="%.4f"),
                            "contribution": st.column_config.NumberColumn("贡献值", format="%+.4f"),
                            "direction": "方向",
                        },
                    )
                    st.caption("贡献值为正表示该特征将预测推向 AI，负值表示推向人类；它不是因果解释。")
                elif explanation is not None:
                    st.info(explanation.get("error", "当前传统模型不支持特征贡献展示。"))
                elif raw_style is not None and scaled_style is not None:
                    render_style_features(raw_style, scaled_style)
                elif selected.kind == "deep":
                    st.info("Encoder-only 模型当前输出分类分数，尚未实现 token 级归因解释。")

            with record_tab:
                render_experiment_record(selected, result)

        except Exception as exc:  # pragma: no cover - Streamlit runtime diagnostics
            st.error(f"模型加载或推理失败：{exc}")
            if "multi_class" in str(exc) or "scikit-learn" in str(exc).lower():
                st.warning(
                    "模型文件与当前 scikit-learn 版本可能不一致。请在训练模型时使用的环境中启动 Demo，"
                    "或在当前环境重新训练对应传统模型。"
                )
            with st.expander("错误详情"):
                st.exception(exc)

if not submitted:
    render_experiment_record(selected, result)

st.divider()
st.warning(
    "使用边界：AI 分数来自当前模型和训练分布，尚未经过面向实际应用的概率校准。"
    "检测结果不应作为处分、定责、学术不端认定或内容下架的唯一依据。"
)
