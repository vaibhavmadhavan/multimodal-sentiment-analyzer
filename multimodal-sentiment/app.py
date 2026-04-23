import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
from fusion import analyse


def interpret_score(score: float) -> str:
    if score >= 0.8:
        return "Very Positive"
    elif score >= 0.4:
        return "Positive"
    elif score >= -0.4:
        return "Neutral"
    elif score >= -0.8:
        return "Negative"
    else:
        return "Very Negative"


def generate_summary(t: dict, f: dict, fu: dict) -> str:
    text_label = interpret_score(t['score'])
    face_label = interpret_score(f['valence_score'])
    alignment_pct = fu['alignment_score'] * 100
    emotion = f.get('dominant_emotion', 'neutral').lower()

    if fu['conflict_detected']:
        return (
            f"The text expresses **{text_label.lower()}** sentiment ({t['score']:+.2f}) "
            f"while the face shows **{emotion}** (valence {f['valence_score']:+.2f}). "
            f"These are in **significant conflict** (alignment: {alignment_pct:.0f}%), "
            f"suggesting the stated sentiment may not reflect how the person truly feels."
        )
    else:
        return (
            f"The text expresses **{text_label.lower()}** sentiment ({t['score']:+.2f}) "
            f"while the face shows **{emotion}** (valence {f['valence_score']:+.2f}). "
            f"These are in **{fu['alignment_label'].lower()}** ({alignment_pct:.0f}%), "
            f"suggesting the feedback is authentic and consistent."
        )


# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Sentiment Analyser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Import a clean sans-serif from Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global font override */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* ── Header ── */
.main-header {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #6C63FF, #48C9B0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    font-family: 'Inter', sans-serif;
}

/* ── Conflict Banner ── */
.conflict-alert {
    background: rgba(255, 183, 77, 0.10);
    border-left: 4px solid #FFB74D;
    padding: 0.85rem 1.1rem;
    border-radius: 8px;
    margin: 1rem 0;
    color: #FFD180;
    font-weight: 500;
    font-size: 0.95rem;
}

/* ── Alignment Banner ── */
.align-alert {
    background: rgba(72, 201, 176, 0.10);
    border-left: 4px solid #48C9B0;
    padding: 0.85rem 1.1rem;
    border-radius: 8px;
    margin: 1rem 0;
    color: #80EAD5;
    font-weight: 500;
    font-size: 0.95rem;
}

/* ── Metric Cards ── */
[data-testid="stMetric"] {
    background: #1C1E2A !important;
    border: 1px solid #2E3050 !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
}

[data-testid="stMetricLabel"] {
    color: #A0A3B1 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

[data-testid="stMetricValue"] {
    color: #E8EAF0 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"] {
    color: #A0A3B1 !important;
    font-size: 0.82rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    color: #A0A3B1 !important;
}

[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom-color: #6C63FF !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #6C63FF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    color: #FFFFFF !important;
    transition: background 180ms ease !important;
}

[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #5A52E0 !important;
}

[data-testid="stButton"] > button[kind="primary"]:disabled {
    background: #2E3050 !important;
    color: #5A5D72 !important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: transparent !important;
    border: 1px solid #2E3050 !important;
    border-radius: 8px !important;
    color: #A0A3B1 !important;
    font-weight: 500 !important;
}

[data-testid="stDownloadButton"] > button:hover {
    border-color: #6C63FF !important;
    color: #6C63FF !important;
}

/* ── Text inputs & text areas ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: #1C1E2A !important;
    border: 1px solid #2E3050 !important;
    border-radius: 8px !important;
    color: #E8EAF0 !important;
}

[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #6C63FF !important;
    box-shadow: 0 0 0 2px rgba(108, 99, 255, 0.20) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1C1E2A !important;
    border: 1px dashed #2E3050 !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr {
    border-color: #2E3050 !important;
    margin: 1.5rem 0 !important;
}

/* ── Warning / Info boxes ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: #6C63FF !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border: 1px solid #2E3050 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #6C63FF !important;
}

/* ── Subheader text ── */
h2, h3 {
    color: #E8EAF0 !important;
    font-weight: 700 !important;
}

/* ── Caption text ── */
[data-testid="stCaptionContainer"] {
    color: #A0A3B1 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-header">🧠 Multi-Modal Sentiment Analyser</p>',
    unsafe_allow_html=True
)
st.caption("Detects alignment between what you **say** and how you **feel** — using NLP + computer vision.")
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📸 Single Analysis", "📋 Batch Analysis", "ℹ️ About"])


# ── Single Analysis ───────────────────────────────────────────────────────────
with tab1:
    col_input, col_results = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("Input")
        text_input = st.text_area(
            "Feedback text",
            placeholder="e.g. This product is great, I really enjoyed using it!",
            height=120
        )
        input_method = st.radio("Image input method", ["📁 Upload", "📷 Webcam"], horizontal=True)

        if input_method == "📁 Upload":
            image_file = st.file_uploader(
                "Face image (JPG or PNG)",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear frontal face photo"
            )
            if image_file:
                st.image(image_file, caption="Uploaded image", use_column_width=True)
        else:
            image_file = st.camera_input("Take a photo")
            if image_file:
                st.image(image_file, caption="Captured photo", use_column_width=True)

        analyse_btn = st.button(
            "🔍 Analyse Sentiment",
            type="primary",
            disabled=(not text_input or not image_file)
        )

    with col_results:
        st.subheader("Results")

        if analyse_btn and text_input and image_file:
            with st.spinner("Running analysis..."):
                pil_image = Image.open(image_file)
                results   = analyse(text_input, pil_image)

            t  = results["text"]
            f  = results["face"]
            fu = results["fusion"]

            if f.get("low_confidence"):
                st.warning("⚠️ Low face detection confidence — facial result may be unreliable. Try a clearer, more frontal photo.")

            # Conflict / Alignment Banner
            if fu["conflict_detected"]:
                st.markdown(f"""
                <div class="conflict-alert">
                ⚠️ <strong>Conflict Detected</strong> — The text sentiment and facial expression
                significantly disagree (difference: {fu['difference']:.2f}).
                </div>""", unsafe_allow_html=True)
            elif fu['alignment_label'] == "Slight Misalignment":
                st.markdown(f"""
                <div class="conflict-alert">
                🔶 <strong>Slight Misalignment</strong> — Text and facial sentiment are somewhat inconsistent
                (difference: {fu['difference']:.2f}).
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="align-alert">
                ✅ <strong>{fu['alignment_label']}</strong> — Text and facial sentiment are consistent.
                </div>""", unsafe_allow_html=True)

            st.markdown(generate_summary(t, f, fu))
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Text Sentiment", f"{t['score']:+.2f}", f"{t['label']} · {t['confidence']*100:.0f}% confident")
            m2.metric("Facial Emotion",  f.get("dominant_emotion", "N/A").upper(),
                                         f"{f['valence_score']:+.2f}")
            m3.metric("Alignment Score", f"{fu['alignment_score'] * 100:.0f}%",
                                         fu["alignment_label"])
            st.caption(f"Text reads as **{interpret_score(t['score'])}** · Face reads as **{interpret_score(f['valence_score'])}**")

            # Alignment Gauge
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fu["alignment_score"] * 100,
                number={"suffix": "%", "font": {"size": 28, "color": "#E8EAF0"}},
                title={"text": "Sentiment Alignment", "font": {"size": 14, "color": "#A0A3B1"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#A0A3B1",
                             "tickfont": {"color": "#A0A3B1"}},
                    "bar":  {"color": "#6C63FF", "thickness": 0.35},
                    "bgcolor": "#1C1E2A",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  40],  "color": "#3D1A1A"},
                        {"range": [40, 70],  "color": "#3D3010"},
                        {"range": [70, 100], "color": "#0D2B25"}
                    ],
                    "threshold": {
                        "line":      {"color": "#6C63FF", "width": 3},
                        "thickness": 0.8,
                        "value":     fu["alignment_score"] * 100
                    }
                }
            ))
            gauge.update_layout(
                height=200,
                margin=dict(t=30, b=10, l=10, r=10),
                paper_bgcolor="#0F1117",
                font_color="#E8EAF0"
            )
            st.plotly_chart(gauge, use_container_width=True)

            # Emotion Distribution Bar Chart
            emotions = f.get("emotion_distribution", {})
            if emotions:
                emo_df = pd.DataFrame({
                    "Emotion":        list(emotions.keys()),
                    "Confidence (%)": list(emotions.values())
                }).sort_values("Confidence (%)", ascending=True)

                fig_bar = px.bar(
                    emo_df,
                    x="Confidence (%)", y="Emotion",
                    orientation="h",
                    title="Facial Emotion Distribution",
                    color="Confidence (%)",
                    color_continuous_scale=[[0, "#1C1E2A"], [0.5, "#6C63FF"], [1.0, "#48C9B0"]]
                )
                fig_bar.update_layout(
                    height=250, showlegend=False,
                    margin=dict(t=35, b=10, l=10, r=10),
                    coloraxis_showscale=False,
                    paper_bgcolor="#0F1117",
                    plot_bgcolor="#1C1E2A",
                    font_color="#E8EAF0",
                    title_font_color="#A0A3B1"
                )
                fig_bar.update_xaxes(gridcolor="#2E3050", zerolinecolor="#2E3050")
                fig_bar.update_yaxes(gridcolor="#2E3050")
                st.plotly_chart(fig_bar, use_container_width=True)

            # Save to History
            st.session_state.history.append({
                "Text (truncated)": text_input[:50] + "...",
                "Text Score":       t["score"],
                "Facial Emotion":   f.get("dominant_emotion", "N/A"),
                "Face Score":       f["valence_score"],
                "Alignment":        fu["alignment_score"],
                "Conflict":         "⚠️ Yes" if fu["conflict_detected"] else "✅ No"
            })


# ── Batch Analysis ────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Batch Analysis")
    st.caption("Upload a CSV with a `review_text` column and image files to analyse multiple entries.")

    csv_file  = st.file_uploader("Upload CSV", type=["csv"], key="csv")
    img_files = st.file_uploader(
        "Upload face images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="imgs"
    )

    if csv_file and img_files and st.button("Run Batch Analysis", type="primary"):
        df      = pd.read_csv(csv_file)
        img_map = {f.name: Image.open(f) for f in img_files}
        results_list = []

        progress = st.progress(0)
        for i, row in df.iterrows():
            text     = str(row.get("review_text", ""))
            img_name = str(row.get("image_file", ""))
            img      = img_map.get(img_name)

            if text and img:
                r = analyse(text, img)
                results_list.append({
                    "Review":     text[:60] + "...",
                    "Text Score": r["text"]["score"],
                    "Emotion":    r["face"]["dominant_emotion"],
                    "Alignment":  r["fusion"]["alignment_score"],
                    "Conflict":   "⚠️" if r["fusion"]["conflict_detected"] else "✅"
                })
            progress.progress((i + 1) / len(df))

        results_df = pd.DataFrame(results_list)

        def colour_rows(row):
            if row["Conflict"] == "⚠️":
                return ["background-color: rgba(255,183,77,0.12); color: #E8EAF0"] * len(row)
            else:
                return ["background-color: rgba(72,201,176,0.10); color: #E8EAF0"] * len(row)

        styled_df = results_df.style.apply(colour_rows, axis=1)
        st.dataframe(styled_df, use_container_width=True)

        # Alignment trend chart
        fig_trend = px.line(
            results_df.reset_index(), x="index", y="Alignment",
            title="Alignment Score Across Reviews",
            markers=True, range_y=[0, 1],
            color_discrete_sequence=["#6C63FF"]
        )
        fig_trend.add_hline(
            y=0.7, line_dash="dash",
            annotation_text="Conflict threshold",
            line_color="#E05C5C",
            annotation_font_color="#E05C5C"
        )
        fig_trend.update_layout(
            paper_bgcolor="#0F1117",
            plot_bgcolor="#1C1E2A",
            font_color="#E8EAF0",
            title_font_color="#A0A3B1"
        )
        fig_trend.update_xaxes(gridcolor="#2E3050", zerolinecolor="#2E3050")
        fig_trend.update_yaxes(gridcolor="#2E3050")
        st.plotly_chart(fig_trend, use_container_width=True)

        # Download button
        csv_out = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV", csv_out,
            file_name="sentiment_results.csv", mime="text/csv"
        )


# ── Session History ───────────────────────────────────────────────────────────
if st.session_state.history:
    st.divider()
    st.subheader("📊 Session History")
    hist_df = pd.DataFrame(st.session_state.history)
    st.dataframe(hist_df, use_container_width=True)

    fig_hist = px.scatter(
        hist_df,
        x="Text Score", y="Face Score",
        color="Alignment", size="Alignment",
        hover_data=["Facial Emotion", "Conflict"],
        title="Text Score vs Face Score (colour = alignment)",
        color_continuous_scale=[[0, "#E05C5C"], [0.5, "#FFB74D"], [1.0, "#48C9B0"]],
        range_x=[-1, 1], range_y=[-1, 1]
    )
    fig_hist.update_layout(
        paper_bgcolor="#0F1117",
        plot_bgcolor="#1C1E2A",
        font_color="#E8EAF0",
        title_font_color="#A0A3B1"
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="#2E3050")
    fig_hist.add_hline(y=0, line_dash="dash", line_color="#2E3050")
    fig_hist.update_xaxes(gridcolor="#2E3050", zerolinecolor="#2E3050")
    fig_hist.update_yaxes(gridcolor="#2E3050")
    st.plotly_chart(fig_hist, use_container_width=True)


# ── About ─────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    This tool simultaneously analyses **text sentiment** and **facial emotion** to detect
    whether what someone says aligns with how they actually feel.

    ---

    ### Architecture
    ```
    Text Input  ──►  DistilBERT (NLP)  ──►  Text Score (-1 to +1)
                                                                    ──►  Fusion Engine  ──►  Alignment Score
    Face Image  ──►  DeepFace CNN      ──►  Face Score (-1 to +1)
    ```

    ### Models
    | Stream | Model | Accuracy |
    |--------|-------|----------|
    | Text | DistilBERT SST-2 | 91.3% on SST-2 |
    | Vision | DeepFace VGG-Face | 96% on LFW |

    ### Fusion Logic
    - **Alignment Score** = `1 - |text_score - face_score| / 2` → range 0–1
    - **Conflict** detected when difference > 0.8
    - **Combined score** = 60% text + 40% face

    ### Tech Stack
    `Python` · `PyTorch` · `HuggingFace Transformers` · `DeepFace` · `Streamlit` · `Plotly`
    """)


    st.markdown("### Emotion → Valence Mapping")
    valence_df = pd.DataFrame({
        "Emotion":  ["Happy", "Surprise", "Neutral", "Fear", "Sad", "Disgust", "Angry"],
        "Valence":  [1.0, 0.3, 0.0, -0.6, -0.7, -0.8, -1.0],
        "Sentiment": ["Very Positive", "Slightly Positive", "Neutral",
                          "Negative", "Negative", "Very Negative", "Very Negative"]
    })
    st.dataframe(valence_df, use_container_width=True, hide_index=True)