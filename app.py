import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="Emotion Recognition",
    page_icon="🎭",
    layout="wide"
)

# =====================
# CONSTANTS
# =====================
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

# =====================
# LOAD MODELS
# =====================
@st.cache_resource
def load_models():
    cnn = tf.keras.models.load_model("models/cnn_base_New.keras")
    resnet = tf.keras.models.load_model("models/resnet50_New.keras")
    mobilenet = tf.keras.models.load_model("models/mobilenetv2_New.keras")
    return cnn, resnet, mobilenet

cnn_model, resnet_model, mobilenet_model = load_models()

# =====================
# IMAGE PREPROCESS
# =====================
def preprocess_image(img):
    if img.mode != "RGB":
        img = img.convert("RGB")

    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

# =====================
# HEADER
# =====================
st.markdown("""
<h1 style='text-align:center;'>🎭 Facial Emotion Recognition</h1>
<p style='text-align:center; font-size:18px;'>
CNN Base • ResNet50 • MobileNetV2
</p>
<hr>
""", unsafe_allow_html=True)

# =====================
# SIDEBAR
# =====================
with st.sidebar:
    st.header("⚙️ Pengaturan Model")
    model_choice = st.radio(
        "Pilih Model",
        ["CNN Base", "ResNet50", "MobileNetV2"]
    )

    st.markdown("---")
    st.caption("📌 Dataset: Facial Emotion Recognition (Kaggle)")
    st.caption("🎓 UAP")

# =====================
# MAIN LAYOUT
# =====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📷 Upload Gambar")
    uploaded_file = st.file_uploader(
        "Masukkan citra wajah",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

with col2:
    st.subheader("Hasil Prediksi")

    if uploaded_file:
        img_array = preprocess_image(image)

        if model_choice == "CNN Base":
            preds = cnn_model.predict(img_array)
        elif model_choice == "ResNet50":
            preds = resnet_model.predict(img_array)
        else:
            preds = mobilenet_model.predict(img_array)

        pred_idx = np.argmax(preds)
        confidence = preds[0][pred_idx] * 100

        # Highlight result
        st.markdown(f"""
        <div style="
            background-color:#f0f2f6;
            padding:20px;
            border-radius:15px;
            text-align:center;
        ">
        <h2> {CLASS_NAMES[pred_idx].upper()}</h2>
        <p style="font-size:20px;">Confidence</p>
        <h3>{confidence:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bar
        st.subheader("📊 Distribusi Probabilitas")
        prob_dict = {
            CLASS_NAMES[i]: float(preds[0][i])
            for i in range(len(CLASS_NAMES))
        }
        st.bar_chart(prob_dict)

    else:
        st.info("⬅️ Silakan upload gambar terlebih dahulu")

# =====================
# FOOTER
# =====================
st.markdown("""
<hr>
<p style='text-align:center; color:gray;'>
Dibuat untuk Ujian Akhir Praktikum 
</p>
""", unsafe_allow_html=True)