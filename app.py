
import os
import io
import re
import streamlit as st
import torch
import timm
import numpy as np
from huggingface_hub import hf_hub_download

from PIL import Image, ImageOps
from torchvision import transforms

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="NourishIQ | Food Freshness Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hugging Face model repository settings.
# Replace this with your actual Hugging Face MODEL repo ID: username/repository-name
HF_REPO_ID = "Kashish-0409/NourishIQ-Food-Freshness-Classifier"
HF_FILENAME = "best_convnextv2.pth"  # exact filename uploaded to Hugging Face

DEVICE = torch.device("cpu")  # Streamlit Community Cloud uses CPU by default


# =====================================================
# PREMIUM UI DESIGN
# =====================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');

.stApp {
    background: #F5F5ED;
    color: #183C2D;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1120px;
    padding: 1.8rem 2rem 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 2.1rem;
    font-weight: 800;
    color: #17452F;
    letter-spacing: -1.3px;
}

.brand span {
    color: #91AC62;
}

.tagline {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: 3px;
    color: #82917D;
    margin-top: -5px;
}

.hero {
    background: linear-gradient(120deg, #153D2D, #285D43);
    padding: 43px 42px;
    border-radius: 25px;
    margin: 24px 0 18px;
    box-shadow: 0 12px 30px rgba(23,65,45,.12);
}

.hero h1 {
    color: #FFFFFF;
    font-family: 'Manrope', sans-serif;
    font-size: clamp(2rem, 4vw, 3.1rem);
    font-weight: 800;
    line-height: 1.12;
    margin: 0 0 14px;
}

.hero p {
    color: #DCE9DB;
    font-size: 1rem;
    max-width: 650px;
    margin: 0;
}

.eyebrow {
    color: #AFC995;
    font-size: .72rem;
    letter-spacing: 2.5px;
    font-weight: 800;
    margin-bottom: 13px;
}

.section-title {
    color: #193D2D;
    font-family: 'Manrope', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    margin: 25px 0 4px;
}

.muted {
    color: #788678;
    font-size: .9rem;
}

.upload-card {
    background: #FFFFFF;
    border: 1px solid #DCE6D7;
    border-radius: 19px;
    padding: 20px;
    margin-top: 14px;
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #DCE6D7;
    border-radius: 18px;
    padding: 19px 17px;
    min-height: 112px;
    box-shadow: 0 5px 18px rgba(30,65,44,.04);
}

.result-label {
    color: #81907D;
    font-size: .76rem;
    font-weight: 600;
    margin-bottom: 8px;
}

.result-value {
    color: #173F2D;
    font-family: 'Manrope', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    overflow-wrap: anywhere;
}

.fresh-status {
    color: #247444;
}

.semi-status {
    color: #B17A27;
}

.rotten-status {
    color: #B34E45;
}

.prediction-row {
    background: #FFFFFF;
    border: 1px solid #E1E8DD;
    border-radius: 13px;
    padding: 13px 15px;
    margin: 10px 0;
}

div[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1px dashed #94AE8D;
    border-radius: 17px;
    padding: 14px;
}

div.stButton > button {
    background: #23583D;
    color: white;
    border: none;
    border-radius: 12px;
    min-height: 49px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: #17432E;
    color: white;
    border: none;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #E1E8DD;
    border-radius: 15px;
    padding: 15px;
}

div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #DCE6D7;
    border-radius: 14px;
}

div[data-testid="stImage"] img {
    border-radius: 16px;
}

.footer {
    text-align: center;
    color: #829080;
    font-size: .8rem;
    padding: 24px 0 8px;
}

hr {
    border-color: #DDE5D8;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# HEADER
# =====================================================

st.markdown("""
<div class="brand">🍃 Nourish<span>IQ</span></div>
<div class="tagline">INTELLIGENT FOOD FRESHNESS</div>

<div class="hero">
    <div class="eyebrow">AI-POWERED FOOD INTELLIGENCE</div>
    <h1>Know your food.<br>Make informed choices.</h1>
    <p>
        Discover visible freshness stages through intelligent
        computer vision. Upload a fruit image and let AI analyse it.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<p class="muted">A computer-vision prototype for preliminary '
    'food freshness assessment. Predictions cannot confirm food '
    'safety or exact shelf life.</p>',
    unsafe_allow_html=True
)


# =====================================================
# MODEL LOADING
# =====================================================

@st.cache_resource
def load_model():

    if HF_REPO_ID.startswith("YOUR_"):
        raise ValueError(
            "Set HF_REPO_ID in app.py to your Hugging Face model repository ID, "
            "for example: username/NourishIQ-Food-Freshness-Classifier."
        )

    try:
        model_file = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="model"
        )
    except Exception as download_error:
        raise RuntimeError(
            f"Could not download {HF_FILENAME} from Hugging Face repo "
            f"{HF_REPO_ID}. Check the repository ID, filename, and visibility. "
            f"Original error: {download_error}"
        ) from download_error

    checkpoint = torch.load(
        model_file,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint["classes"]

    model = timm.create_model(
        "convnextv2_base.fcmae_ft_in22k_in1k",
        pretrained=False,
        num_classes=len(classes)
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(DEVICE)
    model.eval()

    return model, classes


try:
    model, class_names = load_model()

except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()


# =====================================================
# IMAGE PREPROCESSING
# =====================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =====================================================
# LABEL PROCESSING
# =====================================================

def clean_label(label):

    # Remove metadata such as (1-6), (25-35)
    label = re.sub(r"\s*\([^)]*\)", "", str(label))

    return label.replace("_", " ").strip()


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


# =====================================================
# MODEL PERFORMANCE
# =====================================================

with st.expander("📊 View Model Performance"):

    st.markdown("### ConvNeXtV2-Base Evaluation")

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Test Accuracy", "96.07%")
    m2.metric("Macro F1-score", "96.25%")
    m3.metric("Weighted F1-score", "95.93%")
    m4.metric("Test Images", "2,211")

    st.caption(
        "These metrics represent evaluation on the saved test dataset. "
        "They are not the confidence score of an individual prediction."
    )


# =====================================================
# IMAGE UPLOAD
# =====================================================

st.markdown(
    '<div class="section-title">01 · Analyse your food</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="muted">Upload a clear image containing one fruit '
    'for freshness classification.</p>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose your fruit image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG and WEBP."
)


# =====================================================
# IMAGE PREVIEW AND PREDICTION
# =====================================================

if uploaded_file is not None:

    try:
        image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        )

        image = ImageOps.exif_transpose(image).convert("RGB")

        st.markdown(
            '<div class="section-title">02 · Image workspace</div>',
            unsafe_allow_html=True
        )

        left, right = st.columns([1, 1.1], gap="large")

        with left:
            st.image(
                image,
                caption="Uploaded fruit image",
                use_container_width=True
            )

        with right:
            st.markdown("### Ready to analyse?")
            st.markdown(
                '<p class="muted">ConvNeXtV2-Base will analyse '
                'the image and estimate its visible freshness stage.</p>',
                unsafe_allow_html=True
            )

            if st.button(
                "✦  Analyse Freshness",
                type="primary",
                use_container_width=True
            ):

                with st.spinner("Analysing your image..."):

                    tensor = transform(image).unsqueeze(0).to(DEVICE)

                    with torch.inference_mode():
                        logits = model(tensor)
                        probabilities = torch.softmax(logits, dim=1)[0]

                    values, indices = torch.topk(
                        probabilities,
                        k=min(5, len(class_names))
                    )

                    predicted_index = int(indices[0].item())
                    predicted_label = class_names[predicted_index]

                    food, freshness = parse_prediction(
                        predicted_label
                    )

                    confidence = float(values[0].item())

                    st.session_state["nourish_result"] = {
                        "food": food,
                        "freshness": freshness,
                        "confidence": confidence,
                        "indices": indices.cpu().tolist(),
                        "values": values.cpu().tolist()
                    }

    except Exception as error:
        st.error(f"Image processing failed: {error}")


# =====================================================
# RESULTS
# =====================================================

result = st.session_state.get("nourish_result")

if result:

    freshness = result["freshness"]

    status_class = {
        "Fresh": "fresh-status",
        "Semi Fresh": "semi-status",
        "Rotten": "rotten-status"
    }.get(freshness, "")

    st.markdown("---")

    st.markdown(
        '<div class="section-title">03 · Your freshness analysis</div>',
        unsafe_allow_html=True
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">PREDICTED FRUIT</div>'
            f'<div class="result-value">{result["food"] or "Unknown"}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with b:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">FRESHNESS STAGE</div>'
            f'<div class="result-value {status_class}">{freshness}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with c:
        st.markdown(
            f'<div class="result-card">'
            f'<div class="result-label">MODEL CONFIDENCE</div>'
            f'<div class="result-value">{result["confidence"]*100:.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.caption(
        "Confidence is the model's softmax score for the selected class. "
        "It is not a guarantee of prediction correctness."
    )

    st.markdown(
        '<div class="section-title">04 · Top 5 predictions</div>',
        unsafe_allow_html=True
    )

    for index, probability in zip(
        result["indices"],
        result["values"]
    ):

        label = clean_label(class_names[index])
        percent = probability * 100

        st.markdown(
            f'<div class="prediction-row">'
            f'<b>{label}</b>'
            f'<span style="float:right;color:#477A52;font-weight:700">'
            f'{percent:.2f}%</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        st.progress(float(probability))

    st.info(
        "The top five probabilities may not add up to 100%, "
        "because the remaining 19 classes are not displayed."
    )

    st.warning(
        "This is a preliminary image-based estimate. "
        "Do not rely on this prediction alone to decide whether "
        "food is safe to eat."
    )


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    '<div class="footer">'
    '🍃 NourishIQ · AI-Based Food Quality and Freshness Intelligence System'
    '<br>Powered by ConvNeXtV2-Base · 24 Classification Classes'
    '</div>',
    unsafe_allow_html=True
)