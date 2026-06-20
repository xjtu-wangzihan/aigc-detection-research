from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = ROOT/"src"/"aigc_detector"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from data_utils import TEXT_COL, load_dataset  # noqa: E402
from evaluate import positive_scores  # noqa: E402
from explain import explain_linear_text  # noqa: E402
from models import build_model  # noqa: E402
from predict import risk_band  # noqa: E402


@st.cache_resource
def load_or_train_model():
    model_path = ROOT / "models" / "hybrid.joblib"
    if model_path.exists():
        return joblib.load(model_path), "models/hybrid.joblib"
    df = load_dataset(ROOT / "data" / "sample_dataset.csv")
    model = build_model("hybrid")
    model.fit(df[TEXT_COL].tolist(), df["label"].to_numpy())
    return model, "demo-trained sample model"


st.set_page_config(page_title="AIGC 文本检测 Demo", layout="wide")
st.title("AIGC 文本检测 Demo")

model, model_source = load_or_train_model()
st.caption(f"当前模型：{model_source}。样例模型仅用于展示交互，不可作为真实检测结论。")

default_text = "本文围绕网络与信息安全中的AIGC内容鉴别问题展开分析，并从跨域泛化、对抗改写和误报风险三个角度设计实验。"
text = st.text_area("输入待检测文本", value=default_text, height=180)

if st.button("检测", type="primary"):
    score = float(positive_scores(model, [text])[0])
    band = risk_band(score)
    col1, col2, col3 = st.columns(3)
    col1.metric("AI 概率", f"{score:.2%}")
    col2.metric("风险等级", band)
    col3.metric("建议", "人工复核" if band == "uncertain_review_needed" else "记录证据")

    st.progress(min(max(score, 0.0), 1.0))
    explanation = explain_linear_text(model, text)
    if "top_features" in explanation:
        st.subheader("主要贡献特征")
        st.dataframe(pd.DataFrame(explanation["top_features"]), use_container_width=True)
    else:
        st.info(explanation.get("error", "当前模型暂不支持解释。"))

st.divider()
st.subheader("使用提醒")
st.write(
    "AIGC 检测不应作为处分或定责的唯一依据。更稳妥的系统应输出概率、置信区间、证据片段和复核建议，"
    "并在跨域、短文本、非母语写作和改写攻击场景下单独报告误报风险。"
)
