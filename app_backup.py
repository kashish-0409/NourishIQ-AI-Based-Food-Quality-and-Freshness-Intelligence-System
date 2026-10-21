import os
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

st.set_page_config(
    page_title="NourishIQ - Food Freshness AI",
    page_icon="🍎",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "nourishiq_baseline.keras")
LABELS_PATH = os.path.join(BASE_DIR, "models", "class_labels.txt")

st.title("🍎 NourishIQ")
st.subheader("AI-Based Food Quality and Freshness Intelligence System")

st.write(
    "Upload a food image to get a preliminary prediction "
    "of its food category and visible freshness stage."
)

st.info(
    "This is an image-based prediction prototype. "
    "It cannot confirm microbial safety or exact shelf life."
)

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error(f"Class labels file not found: {LABELS_PATH}")
    st.stop()


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def load_labels():
    labels = []
    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            index, name = line.rstrip("\n").split("\t", 1)
            labels.append((int(index), name))

    labels.sort(key=lambda item: item[0])
    return [name for _, name in labels]


try:
    model = load_model()
    class_names = load_labels()
except Exception as error:
    st.error(f"Could not load model or labels: {error}")
    st.stop()


uploaded_file = st.file_uploader(
    "Choose a food image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.image(image, caption="Uploaded food image", use_container_width=True)

        if st.button("Analyse Image", type="primary"):
            with st.spinner("Analysing image..."):
                resized = image.resize((128, 128))
                image_array = np.asarray(resized, dtype=np.float32) / 255.0
                image_array = np.expand_dims(image_array, axis=0)

                probabilities = model.predict(image_array, verbose=0)[0]
                predicted_index = int(np.argmax(probabilities))
                predicted_class = class_names[predicted_index]
                confidence = float(probabilities[predicted_index])

                # Parse freshness stage from the class label
                normalized = predicted_class.replace("_", " ")
                parts = normalized.split()

                if parts[0].lower() == "semi" and len(parts) > 1 and parts[1].lower() == "fresh":
                    freshness = "Semi Fresh"
                    food_name = " ".join(parts[2:])
                elif parts[0].lower() in ("fresh", "rotten"):
                    freshness = parts[0].title()
                    food_name = " ".join(parts[1:])
                else:
                    freshness = "Unclassified"
                    food_name = predicted_class

                # Remove dataset-specific suffixes from the display name
                import re
                food_name = re.sub(r"\s*\([^)]*\)", "", food_name).strip()
                food_name = food_name or predicted_class

                st.success("Analysis complete!")

                col1, col2 = st.columns(2)
                col1.metric("Predicted Food", food_name)
                col2.metric("Predicted Freshness", freshness)

                st.metric("Model Confidence", f"{confidence * 100:.2f}%")
                st.caption(
                    "Confidence is the model's predicted probability for its selected class, "
                    "not a guarantee that the prediction is correct."
                )

                st.subheader("Top 5 Class Predictions")
                top_indices = np.argsort(probabilities)[::-1][:5]

                for index in top_indices:
                    st.write(
                        f"**{class_names[index]}** — "
                        f"{probabilities[index] * 100:.2f}%"
                    )
                    st.progress(float(probabilities[index]))

                st.warning(
                    "Use this result as preliminary guidance only. "
                    "Do not rely on the image prediction alone to decide whether food is safe."
                )

    except Exception as error:
        st.error(f"Could not process this image: {error}")

st.divider()
st.caption("NourishIQ | Minor Project Prototype")
