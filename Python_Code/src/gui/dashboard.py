import streamlit as st
import pandas as pd
import numpy as np
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try importing plotly (interactive charts)
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Custom modules
from src.eda.preprocessing import preprocess_image
from src.models.classifiers import get_model
from src.models.explainability import CustomGradCAM, get_target_layer, overlay_cam_on_image

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Glaucoma AI Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==============================================================================
# PREMIUM DARK-MODE CSS INJECTION
# ==============================================================================
st.markdown("""
<style>
    /* ── GLOBAL DARK THEME ── */
    .stApp {
        background: linear-gradient(135deg, #070b15 0%, #0a0e1a 50%, #07111f 100%);
        color: #d8e4f0;
        font-family: 'Segoe UI', system-ui, sans-serif;
    }
    [data-testid="stHeader"] { background: rgba(7,11,21,0.95); border-bottom: 1px solid rgba(0,212,255,0.12); }

    /* ── HERO BANNER ── */
    .hero-banner {
        background: linear-gradient(135deg, #0b1e35 0%, #0a3050 60%, #0b1e35 100%);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 18px;
        padding: 32px 48px;
        margin-bottom: 28px;
        text-align: center;
        box-shadow: 0 0 60px rgba(0,212,255,0.08), inset 0 1px 0 rgba(255,255,255,0.05);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 900;
        background: linear-gradient(100deg, #00d4ff 0%, #ffffff 50%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0; line-height: 1.2;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #607a9a;
        font-size: 0.92rem;
        margin-top: 10px;
        letter-spacing: 0.3px;
    }
    .hero-author {
        color: #3d5470;
        font-size: 0.8rem;
        margin-top: 6px;
        font-style: italic;
    }

    /* ── METRIC CARDS ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(11,28,50,0.95), rgba(8,16,34,0.95));
        border: 1px solid rgba(0,212,255,0.18);
        border-radius: 14px;
        padding: 22px 20px;
        text-align: center;
        box-shadow: 0 6px 24px rgba(0,0,0,0.35);
        transition: border-color 0.25s, box-shadow 0.25s;
        height: 100%;
    }
    .metric-card:hover {
        border-color: rgba(0,212,255,0.55);
        box-shadow: 0 6px 32px rgba(0,212,255,0.12);
    }
    .metric-value {
        font-size: 2.3rem;
        font-weight: 900;
        color: #00d4ff;
        line-height: 1;
        letter-spacing: -1px;
    }
    .metric-label {
        font-size: 0.72rem;
        color: #607a9a;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-top: 6px;
        font-weight: 600;
    }

    /* ── SECTION HEADERS ── */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #00d4ff;
        border-left: 4px solid #00d4ff;
        padding-left: 14px;
        margin: 28px 0 14px 0;
        letter-spacing: 0.2px;
    }

    /* ── PREDICTION BADGES ── */
    .badge-positive {
        display: block;
        background: linear-gradient(135deg, #c0392b, #e74c3c);
        color: #fff; font-weight: 800; font-size: 0.88rem;
        padding: 9px 0; border-radius: 50px; text-align: center;
        box-shadow: 0 4px 18px rgba(231,76,60,0.45);
        letter-spacing: 0.5px;
    }
    .badge-negative {
        display: block;
        background: linear-gradient(135deg, #1a7a40, #27ae60);
        color: #fff; font-weight: 800; font-size: 0.88rem;
        padding: 9px 0; border-radius: 50px; text-align: center;
        box-shadow: 0 4px 18px rgba(39,174,96,0.45);
        letter-spacing: 0.5px;
    }

    /* ── IMAGE LABEL CHIP ── */
    .img-chip {
        background: rgba(0,212,255,0.08);
        border: 1px solid rgba(0,212,255,0.25);
        border-radius: 8px;
        padding: 6px 12px;
        text-align: center;
        font-weight: 700;
        font-size: 0.82rem;
        margin-bottom: 8px;
    }

    /* ── DIVIDER ── */
    .premium-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,212,255,0.3), transparent);
        margin: 28px 0;
    }

    /* ── TABS ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(8,16,34,0.85);
        border-radius: 14px; padding: 5px; gap: 5px;
        border: 1px solid rgba(0,212,255,0.1);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; color: #607a9a;
        font-weight: 600; padding: 9px 22px; font-size: 0.88rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0a3a58, #083250) !important;
        color: #00d4ff !important;
        border: 1px solid rgba(0,212,255,0.35) !important;
        box-shadow: 0 2px 12px rgba(0,212,255,0.15) !important;
    }

    /* ── FILE UPLOADER ── */
    [data-testid="stFileUploader"] {
        background: rgba(11,28,50,0.6);
        border: 2px dashed rgba(0,212,255,0.28);
        border-radius: 14px; padding: 12px;
    }

    /* ── INFO/WARNING BOXES ── */
    .training-box {
        background: rgba(11,28,50,0.85);
        border: 1px solid rgba(0,212,255,0.2);
        border-radius: 16px; padding: 36px;
        text-align: center;
    }

    /* ── VERDICT CARD ── */
    .verdict-card {
        border-radius: 18px; padding: 28px;
        text-align: center;
    }
    
    /* ── FADE-IN ANIMATION FOR CHART TAB ── */
    .fade-in-chart {
        animation: fadeSlideIn 0.9s ease-out;
    }
    
    @keyframes fadeSlideIn {
        0% {
            opacity: 0;
            transform: translateY(18px);
        }
        100% {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* ── HIDE STREAMLIT BRANDING ── */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# CONSTANTS
# ==============================================================================
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR  = os.path.join(_SCRIPT_DIR, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "glaucoma_classification_benchmark.xlsx")

MODELS       = ['alexnet', 'vgg16', 'densenet121', 'resnet18', 'mobilenet_v2']
MODEL_LABELS = ['AlexNet', 'VGG-16', 'DenseNet-121', 'ResNet-18', 'MobileNetV2']


# PREP_METHODS = ['original', 'he', 'gamma', 'clahe']
# PREP_LABELS = ['Original', 'HE', 'Gamma', 'CLAHE']
# PREP_ICONS        = ['📷']
# PREP_COLORS       = ['#00d4ff']
VIS_PREP_METHODS = ['original', 'he', 'clahe']
VIS_PREP_LABELS = ['Original', 'HE', 'CLAHE']
VIS_PREP_ICONS = ['📷', '📊', '🔬']
VIS_PREP_COLORS = ['#00d4ff', '#a29bfe', '#2ed573']
VIS_PREP_DESCRIPTIONS = [
    "Resize to 224×224, no enhancement",
    "Histogram Equalization for global contrast enhancement",
    "CLAHE for local contrast enhancement"
]
TEST_PREP_METHODS = ['original']
TEST_PREP_LABELS = ['Original']
TEST_PREP_ICONS = ['📷']
TEST_PREP_COLORS = ['#00d4ff']

PREP_DESCRIPTIONS = [
    "Resize to 224×224, no enhancement"
]

MODEL_PARAMS = ['60M', '138M', '7M', '11M', '3.4M']
MODEL_DESCS  = [
    'Classic baseline deep CNN',
    'Deep uniform 3×3 conv architecture',
    'Parameter-efficient dense connections',
    'Residual skip-connection network',
    'Lightweight depthwise-separable convs',
]
MODEL_COLORS = ['#00d4ff', '#a29bfe', '#2ed573', '#ffa502', '#ff6b81']

# Per-database image counts [train, test]
DB_STATS = {
    'BEH':    (461,  154), 'CRFO':   (52,   13),  'DRH':    (80,   30),
    'EyePACS':(2282, 767), 'FIVES':  (271,  96),  'G1020':  (709,  245),
    'HRF':    (21,   13),  'JSIEC':  (32,    8),  'LES':    (19,   12),
    'OIA':    (3103,1047), 'ORIGA':  (650,  226), 'ORIGA2': (452,    0),
    'PAPILA': (305,  84),  'REFUGE': (573,  193), 'SJCHOI': (280,  139),
}

# ==============================================================================
# CHART DATA — MODEL PERFORMANCE
# ==============================================================================
CHART_DATA = pd.DataFrame([
    # Model, Preprocessing, Accuracy, AUC
    ("ResNet18",      "ORIGINAL", 91.3, 0.971),
    ("ResNet18",      "HE",       88.8, 0.953),
    ("ResNet18",      "CLAHE",    89.9, 0.959),

    ("AlexNet",       "ORIGINAL", 89.8, 0.964),
    ("AlexNet",       "HE",       88.7, 0.953),
    ("AlexNet",       "CLAHE",    88.9, 0.956),

    ("DenseNet121",   "ORIGINAL", 91.6, 0.973),
    ("DenseNet121",   "HE",       89.3, 0.960),
    ("DenseNet121",   "CLAHE",    89.6, 0.963),

    ("MobileNetV2",   "ORIGINAL", 90.2, 0.961),
    ("MobileNetV2",   "HE",       89.1, 0.955),
    ("MobileNetV2",   "CLAHE",    89.6, 0.958),

    ("VGG16",         "ORIGINAL", 91.8, 0.973),
    ("VGG16",         "HE",       90.3, 0.963),
    ("VGG16",         "CLAHE",    90.1, 0.964),
], columns=["Model", "Preprocessing", "Accuracy", "AUC"])

# ==============================================================================
# HERO BANNER
# ==============================================================================
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">👁️ Glaucoma AI Classification System</p>
    <p class="hero-sub">
        Deep Learning Benchmark &nbsp;·&nbsp; 3 Preprocessing Methods &nbsp;·&nbsp;
        5 CNN Architectures &nbsp;·&nbsp; 15 Clinical Databases &nbsp;·&nbsp; 20 Experiments
    </p>
    <p class="hero-author">Muhammad Firman Friyadi &nbsp;·&nbsp; LAiMM Lab &nbsp;·&nbsp; NCKU</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# TABS
# ==============================================================================
tab1, tab2, tab3, tab4 , tab5= st.tabs([
    "🏥  Overview & EDA",
    "🔬  Preprocessing",
    "📊  Benchmark Results",
    "🤖  Clinical Tester",
    "📈  Chart",
])

# ==============================================================================
# HELPERS
# ==============================================================================
@st.cache_resource
def load_trained_model(model_name: str, prep_method: str, fold: int = 5):
    exp_name   = f"{model_name}_{prep_method}"
    model_path = os.path.join(RESULTS_DIR, exp_name, f"model_fold{fold}.pth")
    model = get_model(model_name, num_classes=2, pretrained=False)
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location='cpu', weights_only=True)
        )
    else:
        st.warning(f"⚠️ Weights not found: `{model_path}` — using random init.")
    model.eval()
    return model

def run_inference(pil_img: Image.Image, model):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = tf(pil_img).unsqueeze(0)
    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)
        pred  = torch.argmax(out, dim=1).item()
        conf  = probs[0, pred].item()
    return pred, conf, tensor

# ==============================================================================
# TAB 1 — OVERVIEW & EDA
# ==============================================================================
with tab1:

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">📋 Dataset at a Glance</p>', unsafe_allow_html=True)
    kpi = [
        ("12,317", "Total Images"),
        ("9,290",  "Training Set"),
        ("3,027",  "Test Set"),
        ("15",     "Clinical Databases"),
        ("20",     "Experiments"),
    ]
    for col, (val, label) in zip(st.columns(5), kpi):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ── Database Distribution Chart ────────────────────────────────────────────
    st.markdown('<p class="section-header">🗃️ Per-Database Image Distribution</p>', unsafe_allow_html=True)
    db_names = list(DB_STATS.keys())
    train_n  = [v[0] for v in DB_STATS.values()]
    test_n   = [v[1] for v in DB_STATS.values()]

    if PLOTLY_AVAILABLE:
        fig_db = go.Figure()
        fig_db.add_trace(go.Bar(
            name='Train', x=db_names, y=train_n,
            marker_color='#00d4ff', marker_line_color='rgba(0,212,255,0.4)',
            marker_line_width=1, hovertemplate='<b>%{x}</b><br>Train: %{y}<extra></extra>'
        ))
        fig_db.add_trace(go.Bar(
            name='Test', x=db_names, y=test_n,
            marker_color='#ff4757', marker_line_color='rgba(255,71,87,0.4)',
            marker_line_width=1, hovertemplate='<b>%{x}</b><br>Test: %{y}<extra></extra>'
        ))
        fig_db.update_layout(
            barmode='group',
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d8e4f0', family='Segoe UI', size=12),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,212,255,0.2)',
                        borderwidth=1, font_size=12),
            xaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickangle=-35,
                       title='Database', title_font_color='#607a9a'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.04)',
                       title='Image Count', title_font_color='#607a9a'),
            margin=dict(t=20, b=90, l=60, r=20), height=380
        )
        st.plotly_chart(fig_db, use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(14, 5), facecolor='#0a0e1a')
        x = np.arange(len(db_names)); w = 0.4
        ax.bar(x - w/2, train_n, w, color='#00d4ff', label='Train')
        ax.bar(x + w/2, test_n,  w, color='#ff4757', label='Test')
        ax.set_xticks(x); ax.set_xticklabels(db_names, rotation=35, ha='right', color='#d8e4f0')
        ax.set_facecolor('#0d1321'); ax.tick_params(colors='#d8e4f0')
        ax.legend(facecolor='#0d1321', labelcolor='#d8e4f0')
        st.pyplot(fig)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ── Preprocessing Method Cards ──────────────────────────────────────────────
    st.markdown('<p class="section-header">⚙️ Preprocessing Methods</p>', unsafe_allow_html=True)
    for col, icon, label, desc, color in zip(
        st.columns(4), VIS_PREP_ICONS, VIS_PREP_LABELS, VIS_PREP_DESCRIPTIONS, VIS_PREP_COLORS
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:rgba(255,255,255,0.08); text-align:left;">
                <div style="font-size:1.9rem; margin-bottom:10px;">{icon}</div>
                <div style="font-weight:800; color:{color}; font-size:0.97rem;">{label}</div>
                <div style="color:#607a9a; font-size:0.76rem; margin-top:8px; line-height:1.45;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # ── CNN Architecture Cards ──────────────────────────────────────────────────
    st.markdown('<p class="section-header">🧠 CNN Architectures Benchmarked</p>', unsafe_allow_html=True)
    for col, mname, params, desc, color in zip(
        st.columns(5), MODEL_LABELS, MODEL_PARAMS, MODEL_DESCS, MODEL_COLORS
    ):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="border-color:rgba(255,255,255,0.08); text-align:left;">
                <div style="font-weight:800; color:{color}; font-size:0.95rem;">{mname}</div>
                <div style="color:#00d4ff; font-size:0.75rem; margin-top:5px; font-weight:600;">{params} params</div>
                <div style="color:#607a9a; font-size:0.74rem; margin-top:8px; line-height:1.45;">{desc}</div>
            </div>""", unsafe_allow_html=True)

# ==============================================================================
# TAB 2 — PREPROCESSING VISUALIZER
# ==============================================================================
with tab2:
    st.markdown('<p class="section-header">🔬 Fundus Image Preprocessing Visualizer</p>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#607a9a;">Upload a fundus retinal image to see all 3 preprocessing '
        'methods applied side-by-side in real time.</p>',
        unsafe_allow_html=True
    )

    uploaded = st.file_uploader(
        "📂 Drop a fundus image here (JPG · PNG · TIF · BMP)",
        type=["jpg", "jpeg", "png", "tif", "bmp"],
        key="prep_uploader"
    )

    if uploaded is not None:
        tmp = os.path.join(_SCRIPT_DIR, "_tmp_prep.jpg")
        with open(tmp, "wb") as f:
            f.write(uploaded.getbuffer())

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-header">🖼️ Preprocessing Results</p>', unsafe_allow_html=True)

        cols = st.columns(len(VIS_PREP_METHODS))
        # for col, method, label, icon, desc, color in zip(
        #     cols, PREP_METHODS, PREP_LABELS, PREP_ICONS, PREP_DESCRIPTIONS, PREP_COLORS
        # ):
        for col, method, label, icon, desc, color in zip(
            cols, VIS_PREP_METHODS, VIS_PREP_LABELS, VIS_PREP_ICONS, VIS_PREP_DESCRIPTIONS, VIS_PREP_COLORS
        ):
            with col:
                st.markdown(
                    f'<div class="img-chip" style="color:{color}; border-color:rgba(255,255,255,0.12);">'
                    f'{icon} {label}</div>',
                    unsafe_allow_html=True
                )
                img = preprocess_image(tmp, method)
                st.image(img, use_container_width=True)
                st.markdown(
                    f'<p style="color:#607a9a; font-size:0.72rem; text-align:center; '
                    f'margin-top:6px;">{desc}</p>',
                    unsafe_allow_html=True
                )

        if os.path.exists(tmp):
            os.remove(tmp)

# ==============================================================================
# TAB 3 — BENCHMARK RESULTS
# ==============================================================================
with tab3:
    st.markdown('<p class="section-header">📊 Model Benchmark Results</p>', unsafe_allow_html=True)

    if os.path.exists(RESULTS_FILE):
        df_raw = pd.read_excel(RESULTS_FILE)

        mean_df = (
            df_raw
            .groupby(['Model', 'Preprocessing'])[
                ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'AUC']
            ]
            .mean()
            .reset_index()
        )
        mean_df['Experiment'] = mean_df['Model'] + '_' + mean_df['Preprocessing']

        best_row = mean_df.loc[mean_df['AUC'].idxmax()]

        # ── Top KPI row ─────────────────────────────────────────────────────────
        for col, (val, label) in zip(st.columns(4), [
            (f"{best_row['AUC']:.4f}",      "Best AUC"),
            (f"{best_row['Accuracy']:.4f}", "Best Accuracy"),
            (f"{mean_df['Sensitivity'].max():.4f}", "Best Sensitivity"),
            (best_row['Experiment'].upper().replace('_', ' → '), "Best Config"),
        ]):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="font-size:1.5rem;">{val}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        if PLOTLY_AVAILABLE:
            # ── AUC Bar Chart ──────────────────────────────────────────────────
            st.markdown('<p class="section-header">🎯 Mean AUC — All 20 Experiments</p>',
                        unsafe_allow_html=True)
            sorted_df = mean_df.sort_values('AUC', ascending=True)
            fig_auc = px.bar(
                sorted_df, x='Experiment', y='AUC',
                color='Preprocessing',
                color_discrete_sequence=VIS_PREP_COLORS,
                hover_data=['Accuracy', 'Sensitivity', 'Specificity', 'F1_Score'],
                labels={'AUC': 'Mean AUC (5-Fold CV)'}
            )
            fig_auc.add_hline(
                y=0.8, line_dash="dash",
                line_color="rgba(255,255,255,0.25)",
                annotation_text="0.80 reference",
                annotation_font_color="#607a9a"
            )
            fig_auc.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#d8e4f0', family='Segoe UI', size=11),
                legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,212,255,0.15)',
                            borderwidth=1),
                xaxis=dict(gridcolor='rgba(255,255,255,0.04)', tickangle=-40),
                yaxis=dict(gridcolor='rgba(255,255,255,0.04)', range=[0, 1.05]),
                margin=dict(t=20, b=130, l=60, r=20), height=400
            )
            st.plotly_chart(fig_auc, use_container_width=True)

            # ── Radar Chart ────────────────────────────────────────────────────
            st.markdown(
                f'<p class="section-header">🕸️ Performance Radar — Best: '
                f'{best_row["Experiment"].upper()}</p>',
                unsafe_allow_html=True
            )
            radar_metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'F1_Score', 'AUC']
            radar_vals    = [best_row[m] for m in radar_metrics]
            fig_radar = go.Figure(go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself',
                line=dict(color='#00d4ff', width=2),
                fillcolor='rgba(0,212,255,0.12)',
                name=best_row['Experiment']
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1],
                                   gridcolor='rgba(255,255,255,0.08)',
                                   color='#607a9a', tickfont_size=10),
                    angularaxis=dict(gridcolor='rgba(255,255,255,0.08)', color='#d8e4f0'),
                    bgcolor='rgba(0,0,0,0)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#d8e4f0', family='Segoe UI'),
                showlegend=True,
                legend=dict(bgcolor='rgba(0,0,0,0)'),
                margin=dict(t=30, b=30), height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        # ── Full Results Table ─────────────────────────────────────────────────
        st.markdown('<p class="section-header">📋 Full Results (Mean over 5 Folds)</p>',
                    unsafe_allow_html=True)
        disp = mean_df[['Experiment','Accuracy','Sensitivity','Specificity',
                         'Precision','F1_Score','AUC']].copy()
        disp = disp.sort_values('AUC', ascending=False).reset_index(drop=True)
        for c in ['Accuracy','Sensitivity','Specificity','Precision','F1_Score','AUC']:
            disp[c] = disp[c].apply(lambda x: f"{x:.4f}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        # ── CM / ROC Viewer ────────────────────────────────────────────────────
        st.markdown('<p class="section-header">🔍 Detailed Plot Viewer</p>',
                    unsafe_allow_html=True)
        sel_exp = st.selectbox(
            "Select Experiment",
            sorted(mean_df['Experiment'].tolist()),
            format_func=lambda x: x.upper().replace('_', ' → ')
        )
        pc1, pc2 = st.columns(2)
        with pc1:
            cm_path = os.path.join(RESULTS_DIR, sel_exp, "confusion_matrix.png")
            st.markdown('<div class="img-chip">Confusion Matrix</div>', unsafe_allow_html=True)
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Not yet generated.")
        with pc2:
            roc_path = os.path.join(RESULTS_DIR, sel_exp, "roc_curve.png")
            st.markdown('<div class="img-chip">ROC Curve</div>', unsafe_allow_html=True)
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
            else:
                st.info("Not yet generated.")

    else:
        st.markdown("""
        <div class="training-box">
            <div style="font-size:2.8rem; margin-bottom:12px;">⏳</div>
            <div style="color:#00d4ff; font-weight:800; font-size:1.2rem;">Training In Progress</div>
            <div style="color:#607a9a; margin-top:10px; font-size:0.9rem;">
                Results will appear here automatically once
                <code style="color:#00d4ff;">train_and_evaluate.py</code> finishes.
            </div>
        </div>""", unsafe_allow_html=True)

# ==============================================================================
# TAB 4 — INTERACTIVE CLINICAL TESTER
# ==============================================================================
with tab4:
    st.markdown('<p class="section-header">🤖 Interactive Clinical Glaucoma Tester</p>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#607a9a;">Upload a fundus image and select a trained model to get '
        'classification + Grad-CAM across all 3 preprocessing methods — '
        'replicating the MATLAB demo layout.</p>',
        unsafe_allow_html=True
    )

    cfg_col1, cfg_col2, cfg_col3 = st.columns([2, 1, 3])
    with cfg_col1:
        sel_model_idx = st.selectbox(
            "🧠 CNN Model",
            range(len(MODELS)),
            format_func=lambda i: MODEL_LABELS[i]
        )
        sel_model = MODELS[sel_model_idx]
    with cfg_col2:
        sel_fold = st.slider("Fold", 1, 5, 5,
                             help="Which K-Fold checkpoint to use for inference")
    with cfg_col3:
        test_file = st.file_uploader(
            "📂 Upload Fundus Image",
            type=["jpg", "jpeg", "png", "tif", "bmp"],
            key="test_uploader"
        )

    if test_file is not None:
        _, btn_col, _ = st.columns([2, 1, 2])
        with btn_col:
            run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

        if run_btn:
            tmp_test = os.path.join(_SCRIPT_DIR, "_tmp_test.jpg")
            with open(tmp_test, "wb") as f:
                f.write(test_file.getbuffer())

            progress_bar = st.progress(0, text="Initialising...")
            results = {}

            for i, (method, label, icon, color) in enumerate(
                zip(TEST_PREP_METHODS, TEST_PREP_LABELS, TEST_PREP_ICONS, TEST_PREP_COLORS)
            ):
                progress_bar.progress(
                    (i + 1) / len(TEST_PREP_METHODS),
                    text=f"Processing {label}..."
                )
                pil_img = preprocess_image(tmp_test, method=method)
                model   = load_trained_model(sel_model, method, fold=sel_fold)
                pred, conf, tensor = run_inference(pil_img, model)

                cam_overlay = None
                try:
                    target_layer   = get_target_layer(model, sel_model)
                    cam_extractor  = CustomGradCAM(model, target_layer)
                    cam, _, _      = cam_extractor.generate_cam(tensor, target_class=pred)
                    tmp_cam = os.path.join(_SCRIPT_DIR, f"_tmp_cam_{method}.jpg")
                    pil_img.save(tmp_cam)
                    cam_overlay = overlay_cam_on_image(tmp_cam, cam)
                    if os.path.exists(tmp_cam):
                        os.remove(tmp_cam)
                except Exception:
                    pass

                results[method] = dict(
                    image=pil_img, pred=pred,
                    label='POSITIVE' if pred == 1 else 'NEGATIVE',
                    conf=conf, cam=cam_overlay
                )

            progress_bar.empty()
            if os.path.exists(tmp_test):
                os.remove(tmp_test)

            st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<p class="section-header">📋 Results — '
                f'{MODEL_LABELS[sel_model_idx]} · Fold {sel_fold}</p>',
                unsafe_allow_html=True
            )

            # ── ROW 1 : Preprocessed images ────────────────────────────────────
            st.markdown(
                '<p style="color:#607a9a; font-size:0.83rem; margin-bottom:6px;">'
                '▼ Preprocessed Fundus Images & Predictions</p>',
                unsafe_allow_html=True
            )
            img_cols = st.columns(4)
            for col, method, label, icon, color in zip(
                img_cols, TEST_PREP_METHODS, TEST_PREP_LABELS, TEST_PREP_ICONS, TEST_PREP_COLORS
            ):
                r = results[method]
                badge_cls  = "badge-positive" if r['pred'] == 1 else "badge-negative"
                badge_text = "🔴 GLAUCOMA" if r['pred'] == 1 else "🟢 NORMAL"
                with col:
                    st.markdown(
                        f'<div class="img-chip" style="color:{color}; '
                        f'border-color:rgba(255,255,255,0.12);">{icon} {label}</div>',
                        unsafe_allow_html=True
                    )
                    st.image(r['image'], use_container_width=True)
                    st.markdown(f'<div class="{badge_cls}">{badge_text}</div>',
                                unsafe_allow_html=True)
                    st.markdown(
                        f'<p style="text-align:center; color:#607a9a; font-size:0.76rem; '
                        f'margin-top:5px;">Confidence: {r["conf"]*100:.1f}%</p>',
                        unsafe_allow_html=True
                    )
                    st.progress(r['conf'])

            st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

            # ── ROW 2 : Grad-CAM heatmaps ──────────────────────────────────────
            st.markdown(
                '<p style="color:#607a9a; font-size:0.83rem; margin-bottom:6px;">'
                '▼ Grad-CAM Feature Activation Maps (Jet colourmap)</p>',
                unsafe_allow_html=True
            )
            cam_cols = st.columns(4)
            for col, method, label, icon, color in zip(
                cam_cols, TEST_PREP_METHODS, TEST_PREP_LABELS, TEST_PREP_ICONS, TEST_PREP_COLORS
            ):
                r = results[method]
                with col:
                    st.markdown(
                        f'<div class="img-chip" style="color:{color}; '
                        f'border-color:rgba(255,255,255,0.12);">{icon} {label} — Score-CAM</div>',
                        unsafe_allow_html=True
                    )
                    if r['cam'] is not None:
                        st.image(r['cam'], use_container_width=True)
                    else:
                        st.warning("CAM unavailable")

            st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

            # ── Verdict Banner ─────────────────────────────────────────────────
            total_n = len(results)
            pos_n = sum(1 for r in results.values() if r['pred'] == 1)
            neg_n = total_n - pos_n
            if pos_n > 0:
                v_color = "#e74c3c"; v_icon = "🔴"
                v_title = "HIGH RISK — GLAUCOMA DETECTED"
                v_sub   = f"{pos_n}/{total_n} method predicts a POSITIVE classification."
            else:
                v_color = "#27ae60"; v_icon = "🟢"
                v_title = "LOW RISK — LIKELY NORMAL"
                v_sub   = f"{neg_n}/{total_n} method predicts a NEGATIVE classification."

            st.markdown(f"""
            <div style="
                background: linear-gradient(135deg, rgba(11,28,50,0.95), rgba(8,16,34,0.95));
                border: 2px solid {v_color};
                border-radius: 18px;
                padding: 30px 36px;
                text-align: center;
                box-shadow: 0 0 40px {v_color}28;
                margin-top: 8px;">
                <div style="font-size:3rem; line-height:1;">{v_icon}</div>
                <div style="font-size:1.5rem; font-weight:900; color:{v_color};
                            margin: 12px 0 8px 0; letter-spacing:0.5px;">{v_title}</div>
                <div style="color:#d8e4f0; font-size:0.95rem;">{v_sub}</div>
                <div style="color:#3d5470; font-size:0.76rem; margin-top:16px;
                            border-top:1px solid rgba(255,255,255,0.06); padding-top:14px;">
                    ⚠️ This is an AI-assisted research tool only.
                    Always consult a qualified ophthalmologist for clinical diagnosis.
                </div>
            </div>""", unsafe_allow_html=True)

# ==============================================================================
# TAB 5 — CHART
# ==============================================================================
with tab5:
    st.markdown('<div class="fade-in-chart">', unsafe_allow_html=True)

    st.markdown(
        '<p class="section-header">📈 Model & Dataset Performance Chart</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p style="color:#607a9a;">Comparison of validation accuracy and AUC '
        'across five CNN architectures and three preprocessing methods.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    if PLOTLY_AVAILABLE:
        # ── Accuracy Chart ─────────────────────────────────────────────────────
        fig_acc = px.bar(
            CHART_DATA,
            x="Model",
            y="Accuracy",
            color="Preprocessing",
            barmode="group",
            text=CHART_DATA["Accuracy"].apply(lambda x: f"{x:.1f}%"),
            color_discrete_map={
                "ORIGINAL": "#00d4ff",
                "HE": "#ffa502",
                "CLAHE": "#2ed573",
            },
            title="Validation Accuracy by Model & Preprocessing"
        )

        fig_acc.update_traces(
            textposition="outside",
            marker_line_width=1,
            marker_line_color="rgba(255,255,255,0.25)",
        )

        fig_acc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d8e4f0", family="Segoe UI", size=12),
            title=dict(
                font=dict(size=20, color="#d8e4f0"),
                x=0.5,
                xanchor="center"
            ),
            legend=dict(
                title="Preprocessing",
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,212,255,0.18)",
                borderwidth=1,
                font=dict(color="#d8e4f0")
            ),
            xaxis=dict(
                title="Model Architecture",
                gridcolor="rgba(255,255,255,0.04)",
                title_font_color="#607a9a"
            ),
            yaxis=dict(
                title="Accuracy (%)",
                range=[84, 94],
                gridcolor="rgba(255,255,255,0.06)",
                title_font_color="#607a9a"
            ),
            height=460,
            margin=dict(t=70, b=70, l=60, r=30),
            transition=dict(duration=800, easing="cubic-in-out")
        )

        st.plotly_chart(fig_acc, use_container_width=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        # ── AUC Chart ──────────────────────────────────────────────────────────
        fig_auc = px.bar(
            CHART_DATA,
            x="Model",
            y="AUC",
            color="Preprocessing",
            barmode="group",
            text=CHART_DATA["AUC"].apply(lambda x: f"{x:.3f}"),
            color_discrete_map={
                "ORIGINAL": "#00d4ff",
                "HE": "#ffa502",
                "CLAHE": "#2ed573",
            },
            title="Validation AUC by Model & Preprocessing"
        )

        fig_auc.update_traces(
            textposition="outside",
            marker_line_width=1,
            marker_line_color="rgba(255,255,255,0.25)",
        )

        fig_auc.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d8e4f0", family="Segoe UI", size=12),
            title=dict(
                font=dict(size=20, color="#d8e4f0"),
                x=0.5,
                xanchor="center"
            ),
            legend=dict(
                title="Preprocessing",
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,212,255,0.18)",
                borderwidth=1,
                font=dict(color="#d8e4f0")
            ),
            xaxis=dict(
                title="Model Architecture",
                gridcolor="rgba(255,255,255,0.04)",
                title_font_color="#607a9a"
            ),
            yaxis=dict(
                title="Area Under ROC Curve (AUC)",
                range=[0.94, 0.98],
                gridcolor="rgba(255,255,255,0.06)",
                title_font_color="#607a9a"
            ),
            height=460,
            margin=dict(t=70, b=70, l=60, r=30),
            transition=dict(duration=800, easing="cubic-in-out")
        )

        st.plotly_chart(fig_auc, use_container_width=True)

    else:
        # ── Matplotlib fallback ────────────────────────────────────────────────
        st.warning("Plotly is not available. Showing static Matplotlib charts instead.")

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0a0e1a")
        pivot_acc = CHART_DATA.pivot(index="Model", columns="Preprocessing", values="Accuracy")
        pivot_acc.plot(kind="bar", ax=ax)
        ax.set_title("Validation Accuracy by Model & Preprocessing", color="#d8e4f0")
        ax.set_ylabel("Accuracy (%)", color="#d8e4f0")
        ax.set_xlabel("Model Architecture", color="#d8e4f0")
        ax.tick_params(colors="#d8e4f0")
        ax.set_facecolor("#0d1321")
        ax.legend(facecolor="#0d1321", labelcolor="#d8e4f0")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0a0e1a")
        pivot_auc = CHART_DATA.pivot(index="Model", columns="Preprocessing", values="AUC")
        pivot_auc.plot(kind="bar", ax=ax)
        ax.set_title("Validation AUC by Model & Preprocessing", color="#d8e4f0")
        ax.set_ylabel("AUC", color="#d8e4f0")
        ax.set_xlabel("Model Architecture", color="#d8e4f0")
        ax.tick_params(colors="#d8e4f0")
        ax.set_facecolor("#0d1321")
        ax.legend(facecolor="#0d1321", labelcolor="#d8e4f0")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)