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
# CUSTOM CSS (NO WHITE BACKGROUND)
# =====================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #020617;
}

.pred-card {
    padding: 28px;
    border-radius: 20px;
    text-align: center;
    color: white;
    box-shadow: 0px 15px 30px rgba(0,0,0,0.35);
}

.pred-title {
    font-size: 36px;
    font-weight: bold;
}

.pred-conf {
    font-size: 24px;
    margin-top: 10px;
}

hr {
    border: 1px solid #334155;
}

div[data-testid="stFileUploader"] {
    background-color: #020617;
    padding: 15px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# CONSTANTS
# =====================
IMG_SIZE = (224, 224)

CLASS_NAMES = [
    "angry", "disgust", "fear",
    "happy", "neutral", "sad", "surprise"
]

EMOTION_COLOR = {
    "angry": "#ef4444",
    "disgust": "#22c55e",
    "fear": "#8b5cf6",
    "happy": "#facc15",
    "neutral": "#64748b",
    "sad": "#3b82f6",
    "surprise": "#ec4899"
}

EMOJI = {
    "angry": "😠",
    "disgust": "🤢",
    "fear": "😨",
    "happy": "😄",
    "neutral": "😐",
    "sad": "😢",
    "surprise": "😲"
}

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
    st.caption("Dataset: Facial Emotion Recognition")
    st.caption("UAP Praktikum ML")

# =====================
# MAIN LAYOUT
# =====================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Citra Wajah")
    uploaded_file = st.file_uploader(
        "Format JPG / PNG",
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

        emotion = CLASS_NAMES[pred_idx]
        color = EMOTION_COLOR[emotion]
        emoji = EMOJI[emotion]

        # === RESULT CARD ===
        st.markdown(f"""
        <div class="pred-card" style="background:{color};">
            <div class="pred-title">{emoji} {emotion.upper()}</div>
            <div class="pred-conf">Confidence: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # === TOP 3 PREDICTION ===
        st.subheader("Top-3 Prediction")
        top3_idx = preds[0].argsort()[-3:][::-1]

        for i in top3_idx:
            st.progress(float(preds[0][i]), text=f"{CLASS_NAMES[i]} ({preds[0][i]*100:.2f}%)")

    else:
        st.info("Upload gambar terlebih dahulu")

# =====================
# FOOTER
# =====================
st.markdown("""
<hr>
<p style='text-align:center; color:#cbd5f5;'>
Dibuat oleh Ganesha Mahardika Prasetya<br>
Ujian Akhir Praktikum Machine Learning
</p>
""", unsafe_allow_html=True)