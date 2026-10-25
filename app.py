
import os
import re
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="NourishIQ | Food Freshness AI",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "nourishiq_mobilenetv2_finetuned.keras"
)
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_labels.txt")

REPORT_NAMES = [
    "mobilenetv2_finetuned_classification_report.txt",
    "clean_mobilenetv2_classification_report.txt",
]
METRICS_NAMES = [
    "mobilenetv2_finetuned_test_metrics.txt",
    "clean_mobilenetv2_test_metrics.txt",
]
CONFUSION_NAMES = [
    "mobilenetv2_finetuned_confusion_matrix.csv",
    "clean_mobilenetv2_confusion_matrix.csv",
]


def find_file(names):
    """Find a file in the project root or common subfolders."""
    folders = [
        BASE_DIR,
        os.path.join(BASE_DIR, "results"),
        os.path.join(BASE_DIR, "models"),
    ]

    for folder in folders:
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isfile(path):
                return path

    return None


REPORT_PATH = find_file(REPORT_NAMES)
METRICS_PATH = find_file(METRICS_NAMES)
CONFUSION_PATH = find_file(CONFUSION_NAMES)

# ---------------- DESIGN ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
.stApp { background:#F6F5EF; color:#19372B; font-family:'DM Sans',sans-serif; }
.block-container { max-width:1080px; padding-top:2rem; padding-bottom:3rem; }
header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer { visibility:hidden; }
.brand { font-family:'Manrope',sans-serif; font-size:2rem; font-weight:800; color:#173D2D; letter-spacing:-1px; }
.brand span { color:#83A65A; }
.tagline { color:#718174; font-size:.77rem; letter-spacing:2px; font-weight:700; }
.hero { background:linear-gradient(125deg,#163C2D,#285A40); border-radius:25px; padding:42px 38px; color:#FFFFFF; margin:22px 0; }
.hero h1 { font-family:'Manrope',sans-serif; font-size:clamp(2rem,5vw,3.15rem); line-height:1.15; color:#FFFFFF; margin-bottom:14px; }
.hero p { color:#D8E7D8; font-size:1.05rem; max-width:650px; }
.section-title { font-family:'Manrope',sans-serif; color:#19372B; font-size:1.45rem; font-weight:800; margin-top:20px; }
.small-muted { color:#718174; font-size:.92rem; }
div[data-testid="stFileUploader"] { background:#FFFFFF; border:1px dashed #9CB49B; border-radius:18px; padding:16px; }
div.stButton > button { background:#285A40; color:white; border:0; border-radius:12px; padding:.75rem 1.2rem; font-weight:700; min-height:48px; }
div.stButton > button:hover { background:#173D2D; color:white; border:0; }
div[data-testid="stMetric"] { background:white; border:1px solid #E3E9DE; border-radius:17px; padding:20px; }
div[data-testid="stMetricLabel"] { color:#718174; }
div[data-testid="stMetricValue"] { color:#19372B; font-family:'Manrope',sans-serif; font-weight:800; }
div[data-testid="stProgress"] > div { background:#E6EDE2; border-radius:20px; }
div[data-testid="stProgress"] div[role="progressbar"] { background:#80A85B; border-radius:20px; }
div[data-testid="stAlert"] { border-radius:12px; }
hr { border-color:#E0E6DA; }
.footer { text-align:center; color:#718174; font-size:.82rem; padding-top:24px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="brand">🍃 Nourish<span>IQ</span></div>
<div class="tagline">INTELLIGENT FOOD FRESHNESS</div>
<div class="hero">
    <h1>Know your food.<br>Make informed choices.</h1>
    <p>AI-powered image analysis to explore food categories and
    their visible freshness stages in seconds.</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="small-muted">A computer-vision prototype for preliminary '
    'food freshness assessment. Image predictions cannot confirm food safety '
    'or exact shelf life.</div>',
    unsafe_allow_html=True,
)

# ---------------- MODEL AND LABELS ----------------

if not os.path.isfile(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.isfile(LABELS_PATH):
    st.error(f"Class labels file not found: {LABELS_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"preprocess_input": preprocess_input},
        compile=False,
        safe_mode=False,
    )


def load_labels():
    items = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\r\n")
            if not line:
                continue
            index, name = line.split("\t", 1)
            items.append((int(index), name))

    items.sort(key=lambda item: item[0])
    return [name for _, name in items]


def clean_label(label):
    return re.sub(
        r"\s*\([^)]*\)", "", label.replace("_", " ")
    ).strip()


def parse_prediction(label):
    normalized = clean_label(label)
    parts = normalized.split()

    if (
        len(parts) >= 2
        and parts[0].lower() == "semi"
        and parts[1].lower() == "fresh"
    ):
        return " ".join(parts[2:]).strip(), "Semi Fresh"

    if parts and parts[0].lower() in ("fresh", "rotten"):
        return " ".join(parts[1:]).strip(), parts[0].title()

    return normalized, "Unclassified"


try:
    model = load_model()
    class_names = load_labels()

    if model.output_shape[-1] != len(class_names):
        st.error(
            f"Model has {model.output_shape[-1]} outputs, "
            f"but {len(class_names)} labels were loaded."
        )
        st.stop()

except Exception as error:
    st.error(f"Could not load model or class labels: {error}")
    st.stop()

# ---------------- READ EVALUATION METRICS ----------------

def read_model_metrics():
    accuracy = precision = recall = f1_score = support = None

    if REPORT_PATH is not None:
        with open(REPORT_PATH, "r", encoding="utf-8-sig") as file:
            lines = file.readlines()

        for line in lines:
            parts = line.split()
            if not parts:
                continue

            if parts[0].lower() == "accuracy" and len(parts) >= 2:
                try:
                    accuracy = float(parts[1])
                    if len(parts) >= 3 and parts[-1].isdigit():
                        support = int(parts[-1])
                except ValueError:
                    pass

            elif len(parts) >= 5 and parts[:2] == ["macro", "avg"]:
                try:
                    precision = float(parts[2])
                    recall = float(parts[3])
                    f1_score = float(parts[4])
                    if len(parts) >= 6:
                        support = int(parts[5])
                except ValueError:
                    pass

    # Fallback to the saved metrics text file for accuracy.
    if accuracy is None and METRICS_PATH is not None:
        with open(METRICS_PATH, "r", encoding="utf-8-sig") as file:
            metrics_text = file.read()

        match = re.search(
            r"accuracy\s*[:=]\s*([0-9]*\.?[0-9]+)\s*%?",
            metrics_text,
            re.IGNORECASE,
        )

        if match:
            accuracy = float(match.group(1))
            if "%" in match.group(0) or accuracy > 1:
                accuracy /= 100

    return accuracy, precision, recall, f1_score, support



# ---------------- MODEL PERFORMANCE ----------------

with st.expander("View Model Performance", expanded=False):
    st.markdown(
        '<div class="small-muted">Overall model evaluation on the saved test set.</div>',
        unsafe_allow_html=True,
    )

    accuracy, precision, recall, f1_score, support = read_model_metrics()

    if all(
        value is not None
        for value in [accuracy, precision, recall, f1_score]
    ):
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Accuracy", f"{accuracy * 100:.2f}%")
        m2.metric("Precision (Macro)", f"{precision * 100:.2f}%")
        m3.metric("Recall (Macro)", f"{recall * 100:.2f}%")
        m4.metric("F1-score (Macro)", f"{f1_score * 100:.2f}%")

        if support is not None:
            st.caption(f"Evaluation test images: {support:,}")

    else:
        st.warning("Evaluation metrics could not be loaded.")

    if CONFUSION_PATH is not None:
        try:
            confusion_df = pd.read_csv(CONFUSION_PATH, index_col=0)
            st.markdown("**Confusion Matrix**")
            st.dataframe(confusion_df, use_container_width=True)
        except Exception as error:
            st.warning(f"Could not read confusion matrix: {error}")

# ---------------- IMAGE ANALYSIS ----------------

st.markdown(
    '<div class="section-title">Analyse your food</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="small-muted">Upload a clear image of one food item to get started.</p>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a food image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP.",
)

if uploaded_file is not None:
    try:
        image = ImageOps.exif_transpose(
            Image.open(uploaded_file)
        ).convert("RGB")

        left, right = st.columns([1, 1.15], gap="large")

        with left:
            st.image(
                image,
                caption="Your uploaded image",
                use_container_width=True,
            )

        with right:
            st.markdown(
                '<div class="section-title">Ready to analyse?</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<p class="small-muted">Our trained CNN estimates the food '
                'category and visible freshness stage from the image.</p>',
                unsafe_allow_html=True,
            )

            if st.button(
                "✦ Analyse freshness",
                type="primary",
                use_container_width=True,
            ):
                with st.spinner("Analysing your image..."):
                    resized = image.resize((128, 128))
                    array = np.asarray(resized, dtype=np.float32)
                    array = np.expand_dims(array, axis=0)

                    probabilities = np.asarray(
                        model.predict(array, verbose=0)[0]
                    )

                    if len(probabilities) != len(class_names):
                        st.error("Model output and class labels do not match.")
                        st.stop()

                    index = int(np.argmax(probabilities))
                    predicted_label = class_names[index]
                    food, freshness = parse_prediction(predicted_label)
                    confidence = float(probabilities[index])

                    st.session_state["nourish_result"] = {
                        "food": food,
                        "freshness": freshness,
                        "confidence": confidence,
                        "probabilities": probabilities.tolist(),
                    }

    except Exception as error:
        st.error(f"Could not process this image: {error}")

# ---------------- DISPLAY PREDICTION ----------------

result = st.session_state.get("nourish_result")

if result:
    st.markdown("---")
    st.markdown(
        '<div class="section-title">Your analysis</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted food", result["food"] or "Unknown")
    col2.metric("Freshness stage", result["freshness"])
    col3.metric(
        "Model confidence",
        f'{result["confidence"] * 100:.2f}%',
    )

    st.caption(
        "Confidence is the model's score for its selected class; "
        "it is not a guarantee that the prediction is correct."
    )

    st.markdown(
        '<div class="section-title">Top predictions</div>',
        unsafe_allow_html=True,
    )

    probs = np.asarray(result["probabilities"])

    for idx in np.argsort(probs)[::-1][:5]:
        st.markdown(f"**{clean_label(class_names[int(idx)])}**")
        st.progress(
            float(probs[idx]),
            text=f"{probs[idx] * 100:.2f}%",
        )

    st.warning(
        "This is a preliminary image-based assessment. Do not rely on this "
        "prediction alone to decide whether food is safe to eat."
    )

# ---------------- FOOTER ----------------

st.markdown("---")
st.markdown(
    '<div class="footer">NourishIQ · Minor Project Prototype · '
    'AI-assisted visual freshness assessment</div>',
    unsafe_allow_html=True,
)