import streamlit as st
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download
import base64

# Set page configuration
st.set_page_config(
    page_title="AI Pet Classifier",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for macho styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 25%, #16213e 75%, #0f3460 100%);
        color: #ffffff;
    }
    
    .main-header {
        background: linear-gradient(135deg, #000000 0%, #434343 100%);
        border: 2px solid #00ffff;
        padding: 2.5rem;
        border-radius: 0;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(0, 255, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.2), transparent);
        animation: sweep 3s infinite;
    }
    
    @keyframes sweep {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 3.5rem;
        font-weight: 900;
        color: #00ffff;
        text-transform: uppercase;
        letter-spacing: 3px;
        text-shadow: 0 0 20px rgba(0, 255, 255, 0.8);
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        color: #ffffff;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
    }
    
    .upload-section {
        padding: 1rem 0;
        margin: 2rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border: 2px solid #00ff00;
        padding: 2rem;
        border-radius: 0;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 0 25px rgba(0, 255, 0, 0.3);
        position: relative;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #00ff00, #00ffff, #ff0040, #00ff00);
        border-radius: 0;
        z-index: -1;
        animation: borderGlow 2s linear infinite;
    }
    
    @keyframes borderGlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .prediction-text {
        font-family: 'Orbitron', monospace;
        font-size: 3rem;
        font-weight: 900;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 15px currentColor;
    }
    
    .confidence-text {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .cat-result {
        color: #ff0040;
        text-shadow: 0 0 20px #ff0040;
    }
    
    .dog-result {
        color: #00ffff;
        text-shadow: 0 0 20px #00ffff;
    }
    
    .uncertain-result {
        color: #ffaa00;
        text-shadow: 0 0 20px #ffaa00;
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(26, 26, 46, 0.8) 100%);
        border: 1px solid #00ffff;
        padding: 1.5rem;
        border-radius: 0;
        margin: 1rem 0;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.2);
    }
    
    .info-box h3 {
        font-family: 'Orbitron', monospace;
        color: #00ffff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(26, 26, 46, 0.8) 100%);
        border: 1px solid #00ff00;
        border-radius: 0;
        margin: 0 0.5rem;
        flex: 1;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stat-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 25px rgba(0, 255, 0, 0.4);
    }
    
    .emoji-large {
        font-size: 4rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 10px rgba(0, 255, 255, 0.8));
    }
    
    .progress-bar {
        margin: 1rem 0;
    }
    
    /* Streamlit specific overrides */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ff00 0%, #00ffff 100%);
    }
    
    .stFileUploader > div {
        border: 2px solid #00ffff;
        border-radius: 0;
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(26, 26, 46, 0.8) 100%);
    }
    
    .stMarkdown h3 {
        font-family: 'Orbitron', monospace;
        color: #00ffff;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stMarkdown h1, .stMarkdown h2 {
        font-family: 'Orbitron', monospace;
        color: #00ffff;
    }
    
    .stExpanderHeader {
        background: linear-gradient(135deg, rgba(0, 0, 0, 0.8) 0%, rgba(26, 26, 46, 0.8) 100%) !important;
        border: 1px solid #00ffff !important;
        color: #00ffff !important;
        font-family: 'Orbitron', monospace !important;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <div class="main-title">⚡ NEURAL BEAST DETECTOR ⚡</div>
    <div class="subtitle">Military-Grade Vision AI • Cat vs Dog Combat Classification</div>
</div>
""", unsafe_allow_html=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Model loading with enhanced caching
@st.cache_resource
def load_model():
    with st.spinner("🤖 Loading AI Model..."):
        model = timm.create_model("vit_base_patch16_224", pretrained=False)
        model.head = nn.Linear(model.head.in_features, 2)
        model.load_state_dict(torch.load("vit_model.pth", map_location=device))
        model.to(device)
        model.eval()
    return model

# Create columns for layout
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    # Info section
    st.markdown("""
    <div class="info-box">
        <h3>🎯 MISSION BRIEFING:</h3>
        <p>1. Deploy image payload (JPG, JPEG, PNG format)</p>
        <p>2. Neural network will execute tactical analysis via Vision Transformer</p>
        <p>3. Receive classified intel with precision confidence metrics</p>
    </div>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "🚀 DEPLOY TARGET IMAGE",
    type=["jpg", "jpeg", "png"],
    help="Select image payload: JPG, JPEG, or PNG format"
)

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Prediction logic
if uploaded_file is not None:
    try:
        # Create columns for image and results
        img_col, result_col = st.columns([1, 1])
        
        with img_col:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(
                image, 
                caption="🎯 TARGET ACQUIRED", 
                use_container_width=True,
                width=300
            )
        
        with result_col:
            st.success("✅ NEURAL NETWORK ONLINE")
            
            # Processing animation
            with st.spinner("⚡ EXECUTING TACTICAL SCAN..."):
                # Preprocess and predict
                img_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
            
            class_labels = ["🔥 FELINE UNIT", "⚡ CANINE WARRIOR"]
            class_emojis = ["🔥", "⚡"]
            confidence = probabilities[predicted_class].item()
            confidence_threshold = 0.70
            
            # Results display
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if confidence < confidence_threshold:
                st.markdown(f"""
                <div class="emoji-large">⚠️</div>
                <div class="prediction-text uncertain-result">UNIDENTIFIED</div>
                <div class="confidence-text">
                    TARGET CLASSIFICATION INCONCLUSIVE
                </div>
                <div style="margin-top: 1rem; font-size: 1.2rem;">
                    CONFIDENCE: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)
            else:
                result_class = "cat-result" if predicted_class == 0 else "dog-result"
                st.markdown(f"""
                <div class="emoji-large">{class_emojis[predicted_class]}</div>
                <div class="prediction-text {result_class}">{class_labels[predicted_class]}</div>
                <div class="confidence-text">CONFIDENCE: {confidence:.1%}</div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Progress bars for both classes
            st.markdown("### 📊 TACTICAL ANALYSIS:")
            
            cat_prob = probabilities[0].item()
            dog_prob = probabilities[1].item()
            
            st.markdown("🔥 **FELINE UNIT:**")
            st.progress(cat_prob)
            st.markdown(f"**{cat_prob:.1%}**")
            
            st.markdown("⚡ **CANINE WARRIOR:**")
            st.progress(dog_prob)
            st.markdown(f"**{dog_prob:.1%}**")
    
    except UnidentifiedImageError:
        st.error("❌ File yang diunggah bukan gambar yang valid atau rusak.")
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan yang tidak terduga: {e}")

# Footer with stats
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-item">
        <div style="font-size: 2rem;">🤖</div>
        <div style="font-weight: bold;">Vision Transformer</div>
        <div>State-of-the-art AI</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-item">
        <div style="font-size: 2rem;">⚡</div>
        <div style="font-weight: bold;">Real-time</div>
        <div>Instant Classification</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-item">
        <div style="font-size: 2rem;">🎯</div>
        <div style="font-weight: bold;">High Accuracy</div>
        <div>Reliable Results</div>
    </div>
    """, unsafe_allow_html=True)

# Additional info
with st.expander("ℹ️ Tentang Model"):
    st.markdown("""
    **Vision Transformer (ViT)** adalah arsitektur deep learning yang menggunakan mekanisme attention 
    untuk mengklasifikasikan gambar. Model ini telah dilatih khusus untuk membedakan antara gambar 
    kucing dan anjing dengan akurasi tinggi.
    
    **Fitur:**
    - Menggunakan ViT base patch16_224
    - Preprocessing otomatis
    - Confidence threshold untuk hasil yang lebih akurat
    - Interface yang user-friendly
    """)

st.markdown("""
<div style="text-align: center; margin-top: 2rem; opacity: 0.7;">
    Made with ❤️ using Streamlit & PyTorch
</div>
""", unsafe_allow_html=True)
