#!/usr/bin/env python3
"""DPO demo — base vs tuned, live (Streamlit Community Cloud).

Type a warehouse-employee review; the base Qwen2.5-0.5B-Instruct and the
LoRA-DPO-tuned adapter label it side by side with the project's v2 prompt
(the exact setup of the frozen evaluation: k_shot = 0, greedy, 128 tokens).

Memory design for the free tier's ~2.7 GB RAM: ONE shared base model in
float16 (~1 GB), wrapped once by the LoRA adapter. Base behavior comes from
`disable_adapter()` — no second weight copy. Generation runs on CPU only.

Honest scope: dpo on silver preferences — not RLHF, no reward model, no
human preference data. Agreement with an llm consensus labeler, not
accuracy. Measured on 27 review-disjoint held-out pairs: risk-class
agreement 40.7% -> 74.1%, sentiment unchanged, theme F1 within noise.
"""

import json
import re
from pathlib import Path

import streamlit as st
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
# committed to the repo — data_local/ does not deploy
ADAPTER_DIR = str(Path(__file__).resolve().parent / "adapter")

st.set_page_config(page_title="reviewinsight dpo demo", page_icon="🏷️", layout="wide")

# ---- fieldnotes theme, matched to the portfolio site ----------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Source+Sans+3:wght@400;500;600&family=Spectral:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', -apple-system, sans-serif;
        background: #f5f1e8;
        color: #25251f;
    }
    .stApp, [data-testid="stMarkdownContainer"] { text-transform: lowercase; }
    input, textarea { text-transform: none; }

    .eyebrow {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        letter-spacing: 0.22em; color: #a43b36;
    }
    .main-title {
        font-family: 'Spectral', Georgia, serif; font-weight: 500;
        font-size: 2rem; color: #25251f; margin: 0.3rem 0 0.5rem;
    }
    .lede { color: #52514e; font-size: 0.98rem; }
    .honesty {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem;
        color: #482728; border: 1px dashed #482728; background: #fbf9f4;
        padding: 0.6rem 0.85rem; margin: 0.8rem 0; max-width: 700px;
    }
    .panel-name {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
        letter-spacing: 0.1em; color: #52514e; margin-bottom: 0.6rem;
    }
    .stButton > button {
        border-radius: 2px; font-family: 'IBM Plex Mono', monospace;
        background: #a43b36; border: 1px solid #a43b36; color: #fbf9f4;
    }
    .stButton > button:hover { background: #482728; border-color: #482728; }
    [data-baseweb="textarea"], .stTextArea textarea {
        background: #fbf9f4; border: 1px solid #d8d1c4; border-radius: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---- the v2 "Role Context Enhanced" template, verbatim from src/prompts ----
SYSTEM_PROMPT = """You are an HR analyst specializing in employee retention at Amazon fulfillment centers and similar warehouse operations.

You understand industry-specific terminology:
- VTO: Voluntary Time Off
- MET: Mandatory Extra Time (overtime)
- Peak Season: High-volume periods (November-December, Prime Day)
- FC: Fulfillment Center
- Rate/Quota: Items processed per hour

When analyzing sentiment, consider:
- Overall tone, not just individual complaints
- Severity of issues raised
- Whether complaints are deal-breakers or minor annoyances"""

USER_TEMPLATE = """Analyze this employee review:

Review: {review_text}

Extract:
1. Sentiment (1=Very Negative, 2=Negative, 3=Neutral, 4=Positive, 5=Very Positive)
2. Main themes (select up to 3: overtime, pay_benefits, management, safety, career_growth, workload, work_life_balance, training, culture, other)
3. Retention risk (low/medium/high based on severity of complaints)

Return JSON only, no other text:
{{
  "sentiment": <1-5>,
  "themes": ["theme1", "theme2", "theme3"],
  "retention_risk": "low|medium|high"
}}"""

MAX_NEW_TOKENS = 128


@st.cache_resource(show_spinner="loading the base model (~1 gb, first visit only)…")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    tuned = PeftModel.from_pretrained(model, ADAPTER_DIR)
    tuned.eval()
    torch.set_num_threads(4)
    return tokenizer, tuned


def generate_label(model, tokenizer, review_text: str, use_adapter: bool) -> str:
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": USER_TEMPLATE.format(review_text=review_text)}],
        tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        if use_adapter:
            out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                 do_sample=False, pad_token_id=tokenizer.eos_token_id)
        else:
            with model.disable_adapter():
                out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                     do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)


def parse_label(raw: str):
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        label = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    sentiment, themes, risk = label.get("sentiment"), label.get("themes"), label.get("retention_risk")
    if not isinstance(sentiment, int) or not isinstance(themes, list):
        return None
    return {"sentiment": sentiment,
            "themes": [str(t) for t in themes][:3],
            "retention_risk": str(risk) if risk else "?"}


def render_panel(label, raw: str, note: str) -> str:
    if label is None:
        return (f"<div style='font-family:IBM Plex Mono,monospace;font-size:12px;color:#77776c'>"
                f"no parsable label — raw output:</div>"
                f"<pre style='white-space:pre-wrap;font-size:11.5px;color:#52514e'>{raw[:400]}</pre>")
    risk_color = {"low": "#89927a", "medium": "#b86a62", "high": "#a43b36"}.get(
        label["retention_risk"], "#77776c")
    stars = "★" * label["sentiment"] + "☆" * (5 - label["sentiment"])
    chips = "".join(
        f"<span style='display:inline-block;border:1px solid #d8d1c4;background:#fbf9f4;"
        f"padding:2px 10px;margin:2px;font-family:IBM Plex Mono,monospace;"
        f"font-size:12.5px;color:#52514e'>{t}</span>" for t in label["themes"])
    return (f"<div style='font-size:22px;color:#25251f'>{stars} "
            f"<span style='font-size:14px;color:#77776c'>({label['sentiment']}/5)</span></div>"
            f"<div style='margin:8px 0'>{chips}</div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:13px;"
            f"color:{risk_color};font-weight:500'>retention risk: {label['retention_risk']}</div>"
            f"<div style='font-family:IBM Plex Mono,monospace;font-size:11px;"
            f"color:#77776c;margin-top:8px'>{note}</div>")


# ---- page ------------------------------------------------------------------
st.markdown("<p class='eyebrow'>reviewinsight ai · the dpo extension, live</p>", unsafe_allow_html=True)
st.markdown("<h1 class='main-title'>base vs dpo-tuned — label a review, watch the alignment shift</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='lede'>the 525 cross-prompt silver labels became 137 constructed preference "
            "pairs (consensus = chosen, outlier = rejected), split 110/27 by review id, and a lora "
            "adapter was tuned on qwen2.5-0.5b-instruct. measured on the 27 review-disjoint held-out "
            "pairs: risk-class agreement 40.7% → 74.1% (+33.3 pp), sentiment exactly unchanged, "
            "theme f1 within noise. on any single review the two models may agree — the measured "
            "shift is distributional.</p>", unsafe_allow_html=True)
st.markdown("<div class='honesty'>dpo on silver preferences — not rlhf: no reward model, no human "
            "preference data. agreement with an llm consensus labeler, not accuracy · n = 27 → "
            "roughly ±18 pp · one seed, one run</div>", unsafe_allow_html=True)

review = st.text_area(
    "warehouse employee review",
    placeholder="Example: The benefits are good but the mandatory overtime is exhausting...",
    height=120, label_visibility="collapsed")

if st.button("label it — base vs tuned", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("type a review first.")
    else:
        tokenizer, tuned = load_model()
        col1, col2 = st.columns(2)
        with st.spinner("base model labeling… (cpu, may take up to a minute)"):
            base_raw = generate_label(tuned, tokenizer, review, use_adapter=False)
        with col1:
            st.markdown("<p class='panel-name'>BASE MODEL</p>", unsafe_allow_html=True)
            st.markdown(render_panel(parse_label(base_raw), base_raw,
                                     "base · qwen2.5-0.5b-instruct"), unsafe_allow_html=True)
        with st.spinner("dpo-tuned model labeling…"):
            tuned_raw = generate_label(tuned, tokenizer, review, use_adapter=True)
        with col2:
            st.markdown("<p class='panel-name'>LORA-DPO TUNED</p>", unsafe_allow_html=True)
            st.markdown(render_panel(parse_label(tuned_raw), tuned_raw,
                                     "lora-dpo tuned · adapter from the reviewinsight run"),
                        unsafe_allow_html=True)

st.markdown("<p style='font-family:IBM Plex Mono,monospace;font-size:11.5px;color:#77776c'>"
            "both runs use the project's v2 prompt (k_shot = 0, greedy, 128 tokens) — the exact "
            "setup of the frozen evaluation. outputs are raw model behavior. "
            "<a href='https://github.com/oleeveeuh/ReviewInsightAI' style='color:#a43b36'>repository ↗</a> · "
            "<a href='https://github.com/oleeveeuh/ReviewInsightAI/blob/main/docs/ALIGNMENT.md' "
            "style='color:#a43b36'>alignment docs ↗</a> · "
            "<a href='https://oleeveeuh.github.io/' style='color:#a43b36'>portfolio ↗</a></p>",
            unsafe_allow_html=True)
