import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import plotly.graph_objects as go

st.set_page_config(
    page_title="EyeScan AI — Eye Disease Classifier",
    page_icon="👁️",
    layout="wide",
)

CLASS_NAMES = ["Cataract", "Diabetic Retinopathy", "Glaucoma", "Normal"]

CLASS_INFO = {
    "Cataract": {"icon": "🫧", "color": "#F4A261", "desc": "Clouding of the eye's natural lens, leading to blurry vision."},
    "Diabetic Retinopathy": {"icon": "🩸", "color": "#E76F51", "desc": "Diabetes-related damage to blood vessels in the retina."},
    "Glaucoma": {"icon": "⚡", "color": "#9B5DE5", "desc": "Increased eye pressure causing damage to the optic nerve."},
    "Normal": {"icon": "✅", "color": "#2A9D8F", "desc": "No signs of disease detected. Eye appears healthy."},
}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    background-color: #060B18;
    color: #E8EAF0;
}
.stApp {
    background: linear-gradient(135deg, #060B18 0%, #0D1B2A 100%);
}
.block-container {
    padding: 2rem 4rem !important;
    max-width: 100% !important;
}
.hero-wrapper {
    text-align: center;
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid rgba(100,223,223,0.1);
    margin-bottom: 2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(100,223,223,0.08);
    color: #64DFDF;
    border: 1px solid rgba(100,223,223,0.3);
    border-radius: 999px;
    padding: 0.4rem 1.4rem;
    font-size: 0.8rem;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif !important;
    font-size: 5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #64DFDF, #48CAE4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.5rem 0;
    letter-spacing: -2px;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    font-size: 1.15rem;
    color: #7A8BA0;
    font-weight: 400;
    line-height: 1.7;
    max-width: 700px;
    margin: 0 auto;
}
.stats-row {
    display: flex;
    justify-content: center;
    gap: 4rem;
    margin: 2rem auto;
    padding: 1.5rem 3rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    max-width: 700px;
}
.stat-item { text-align: center; }
.stat-number {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: #64DFDF;
    line-height: 1.1;
}
.stat-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #7A8BA0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 0.2rem;
}
.upload-section {
    max-width: 700px;
    margin: 0 auto 2rem auto;
}
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #BCC5D0;
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(100,223,223,0.25) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
    background: rgba(255,255,255,0.01) !important;
}
.result-card {
    border-radius: 20px;
    padding: 2rem 2.5rem;
    border: 1px solid rgba(255,255,255,0.08);
    background: rgba(255,255,255,0.03);
    height: 100%;
}
.result-icon { font-size: 3.5rem; margin-bottom: 0.5rem; }
.result-disease {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
    line-height: 1.1;
}
.result-desc {
    font-family: 'Inter', sans-serif;
    font-size: 1rem;
    color: #7A8BA0;
    margin-top: 0.5rem;
    line-height: 1.6;
}
.section-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: #7A8BA0;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin: 1.5rem 0 0.4rem 0;
}
.confidence-num {
    font-family: 'Syne', sans-serif !important;
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    letter-spacing: -2px;
    line-height: 1;
}
.bar-container {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 10px;
    margin: 0.6rem 0 1.5rem 0;
    overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 999px; }
.class-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
}
.class-name {
    font-family: 'Inter', sans-serif;
    font-size: 0.95rem;
    color: #BCC5D0;
    font-weight: 500;
}
.class-pct {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #64DFDF;
}
.chart-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #E8EAF0;
    margin: 1rem 0 0.5rem 0;
}
.disclaimer {
    font-family: 'Inter', sans-serif;
    font-size: 0.8rem;
    color: #445566;
    text-align: center;
    margin-top: 3rem;
    padding: 1.5rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if not os.path.exists("eye_disease_model.h5"):
        return None
    return tf.keras.models.load_model("eye_disease_model.h5")

def preprocess_image(img, target_size=(224, 224)):
    img = img.convert("RGB").resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# Hero
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">KaggleHacX '26 — AI for Healthcare</div>
    <div class="hero-title">👁️ EyeScan AI</div>
    <div class="hero-sub">
        Advanced deep learning system for early detection of retinal diseases.<br>
        Upload a fundus image and get instant AI-powered diagnosis with confidence scores.
    </div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-number">92%</div>
        <div class="stat-label">Model Accuracy</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">4</div>
        <div class="stat-label">Disease Classes</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">4,217</div>
        <div class="stat-label">Training Images</div>
    </div>
    <div class="stat-item">
        <div class="stat-number">99%</div>
        <div class="stat-label">Best Class F1</div>
    </div>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.warning("⚠️ Model file `eye_disease_model.h5` not found.")

# Upload
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<div class="upload-label">Upload Retinal Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file and model:
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 0.05, 1], gap="small")

    with col1:
        st.image(image, use_container_width=True, caption="Uploaded Retinal Image")

    with col3:
        with st.spinner("Analyzing retinal image..."):
            processed = preprocess_image(image)
            preds = model.predict(processed)[0]

        top_idx = int(np.argmax(preds))
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(preds[top_idx]) * 100
        info = CLASS_INFO[top_class]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-icon">{info['icon']}</div>
            <div class="result-disease" style="color:{info['color']};">{top_class}</div>
            <div class="result-desc">{info['desc']}</div>
            <div class="section-label">Confidence Score</div>
            <div class="confidence-num" style="color:{info['color']};">{top_conf:.1f}%</div>
            <div class="bar-container">
                <div class="bar-fill" style="width:{top_conf:.1f}%; background:linear-gradient(90deg,{info['color']},#64DFDF);"></div>
            </div>
            <div class="section-label">All Classes</div>
        """, unsafe_allow_html=True)

        sorted_idx = np.argsort(preds)[::-1]
        for i in sorted_idx:
            cname = CLASS_NAMES[i]
            cpct = float(preds[i]) * 100
            cinfo = CLASS_INFO[cname]
            st.markdown(f"""
            <div class="class-row">
                <span class="class-name">{cname}</span>
                <span class="class-pct">{cpct:.1f}%</span>
            </div>
            <div class="bar-container" style="height:6px; margin:2px 0 8px 0;">
                <div class="bar-fill" style="width:{cpct:.1f}%; background:{cinfo['color']}; opacity:0.7;"></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # Charts Row
    st.markdown("---")
    chart_col1, chart_col2 = st.columns([1, 1], gap="large")

    with chart_col1:
        st.markdown('<div class="chart-title">📊 Prediction Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(data=[go.Pie(
            labels=CLASS_NAMES,
            values=[float(p)*100 for p in preds],
            hole=0.55,
            marker=dict(colors=["#F4A261","#E76F51","#9B5DE5","#2A9D8F"]),
            textinfo='label+percent',
            textfont=dict(size=14, color="white"),
        )])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter', size=13),
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
            height=380,
            annotations=[dict(
                text=f"<b>{top_conf:.0f}%</b><br>{top_class}",
                x=0.5, y=0.5,
                font=dict(size=18, color=info['color'], family='Syne'),
                showarrow=False
            )]
        )
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        st.markdown('<div class="chart-title">📈 Class Confidence Scores</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=[float(p)*100 for p in preds],
            y=CLASS_NAMES,
            orientation='h',
            marker=dict(
                color=["#F4A261","#E76F51","#9B5DE5","#2A9D8F"],
                line=dict(width=0)
            ),
            text=[f"{float(p)*100:.1f}%" for p in preds],
            textposition='outside',
            textfont=dict(color='white', size=14, family='Inter')
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter', size=13),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[0, 120]
            ),
            yaxis=dict(
                showgrid=False,
                tickfont=dict(size=14, family='Inter'),
                categoryorder='total ascending'
            ),
            margin=dict(t=20, b=20, l=10, r=80),
            height=380,
            bargap=0.35,
        )
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
<div class="disclaimer">
    ⚕️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only.
    Not a substitute for professional medical diagnosis. Always consult a qualified ophthalmologist.
</div>
""", unsafe_allow_html=True)