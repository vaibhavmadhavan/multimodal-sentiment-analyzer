import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from PIL import Image
from fusion import analyse

# ── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Modal Sentiment Analyser",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
.conflict-alert {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
.align-alert {
    background: #d4edda;
    border-left: 4px solid #28a745;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-header">🧠 Multi-Modal Sentiment Analyser</p>',
    unsafe_allow_html=True
)
st.caption("Detects alignment between what you **say** and how you **feel** — using NLP + computer vision.")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📸 Single Analysis", "📋 Batch Analysis"])

# ──────────────────────────────────────────────────────────────────
# TAB 1: Single Analysis
# ──────────────────────────────────────────────────────────────────
with tab1:
    col_input, col_results = st.columns([1, 1.5], gap="large")

    with col_input:
        st.subheader("Input")
        text_input = st.text_area(
            "Feedback text",
            placeholder="e.g. This product is great, I really enjoyed using it!",
            height=120
        )
        image_file = st.file_uploader(
            "Face image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear frontal face photo"
        )
        if image_file:
            st.image(image_file, caption="Uploaded image", use_column_width=True)

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

            # ── Conflict / Alignment Banner ──────────────────────
            if fu["conflict_detected"]:
                st.markdown(f"""
                <div class="conflict-alert">
                ⚠️ <strong>Conflict Detected</strong> — The text sentiment and facial expression
                significantly disagree (difference: {fu['difference']:.2f}).
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="align-alert">
                ✅ <strong>{fu['alignment_label']}</strong> — Text and facial sentiment are consistent.
                </div>""", unsafe_allow_html=True)

            # ── Metrics Row ──────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Text Sentiment",  f"{t['score']:+.2f}",  t["label"])
            m2.metric("Facial Emotion",  f.get("dominant_emotion", "N/A").upper(),
                                         f"{f['valence_score']:+.2f}")
            m3.metric("Alignment Score", f"{fu['alignment_score'] * 100:.0f}%",
                                         fu["alignment_label"])

            # ── Alignment Gauge ───────────────────────────────────
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fu["alignment_score"] * 100,
                number={"suffix": "%", "font": {"size": 28}},
                title={"text": "Sentiment Alignment", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": "#0d6efd", "thickness": 0.3},
                    "steps": [
                        {"range": [0,  40],  "color": "#ffcccc"},
                        {"range": [40, 70],  "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#d4edda"}
                    ],
                    "threshold": {
                        "line":      {"color": "black", "width": 2},
                        "thickness": 0.75,
                        "value":     fu["alignment_score"] * 100
                    }
                }
            ))
            gauge.update_layout(height=200, margin=dict(t=30, b=10, l=10, r=10))
            st.plotly_chart(gauge, use_container_width=True)

            # ── Emotion Distribution Bar Chart ────────────────────
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
                    color_continuous_scale="blues"
                )
                fig_bar.update_layout(
                    height=250, showlegend=False,
                    margin=dict(t=35, b=10, l=10, r=10),
                    coloraxis_showscale=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Save to History ───────────────────────────────────
            st.session_state.history.append({
                "Text (truncated)": text_input[:50] + "...",
                "Text Score":       t["score"],
                "Facial Emotion":   f.get("dominant_emotion", "N/A"),
                "Face Score":       f["valence_score"],
                "Alignment":        fu["alignment_score"],
                "Conflict":         "⚠️ Yes" if fu["conflict_detected"] else "✅ No"
            })

# ──────────────────────────────────────────────────────────────────
# TAB 2: Batch Analysis
# ──────────────────────────────────────────────────────────────────
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
        st.dataframe(results_df, use_container_width=True)

        # Alignment trend chart
        fig_trend = px.line(
            results_df.reset_index(), x="index", y="Alignment",
            title="Alignment Score Across Reviews",
            markers=True, range_y=[0, 1]
        )
        fig_trend.add_hline(
            y=0.7, line_dash="dash",
            annotation_text="Conflict threshold",
            line_color="red"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        # Download button
        csv_out = results_df.to_csv(index=False)
        st.download_button(
            "📥 Download Results CSV", csv_out,
            file_name="sentiment_results.csv", mime="text/csv"
        )

# ──────────────────────────────────────────────────────────────────
# HISTORY LOG (bottom of page)
# ──────────────────────────────────────────────────────────────────
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
        color_continuous_scale="RdYlGn",
        range_x=[-1, 1], range_y=[-1, 1]
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_hist.add_hline(y=0, line_dash="dash", line_color="gray")
    st.plotly_chart(fig_hist, use_container_width=True)