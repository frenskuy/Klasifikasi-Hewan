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

# Custom CSS for cyberpunk styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700&family=Rajdhani:wght@300;400;500;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 25%, #16213e 50%, #0f1419 100%);
        color: #e0e0e0;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f1419 100%);
        border: 1px solid rgba(0, 255, 255, 0.3);
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 255, 255, 0.1);
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
        background: linear-gradient(90deg, transparent, rgba(0, 255, 255, 0.1), transparent);
        animation: sweep 4s infinite;
    }
    
    @keyframes sweep {
        0% { left: -100%; }
        100% { left: 100%; }
    }
    
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.2rem;
        font-weight: 600;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }
    
    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0;
        position: relative;
        z-index: 1;
    }
    
    .upload-section {
        padding: 1rem 0;
        margin: 2rem 0;
    }
    
    .result-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f1419 100%);
        border: 1px solid rgba(0, 255, 170, 0.4);
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 255, 170, 0.1);
        position: relative;
    }
    
    .prediction-text {
        font-family: 'Orbitron', monospace;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 0 0 8px currentColor;
    }
    
    .confidence-text {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.2rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.9;
    }
    
    .cat-result {
        color: #ff6b9d;
        text-shadow: 0 0 10px rgba(255, 107, 157, 0.5);
    }
    
    .dog-result {
        color: #00d4ff;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .uncertain-result {
        color: #ffaa44;
        text-shadow: 0 0 10px rgba(255, 170, 68, 0.5);
    }
    
    .info-box {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.6) 0%, rgba(22, 33, 62, 0.6) 100%);
        border: 1px solid rgba(0, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 6px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(0, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .info-box h3 {
        font-family: 'Orbitron', monospace;
        color: #00ffaa;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 1.1rem;
        margin-bottom: 1rem;
    }
    
    .info-box p {
        font-family: 'Rajdhani', sans-serif;
        color: #c0c0c0;
        font-size: 1rem;
        line-height: 1.5;
        margin-bottom: 0.5rem;
    }
    
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
    }
    
    .stat-item {
        text-align: center;
        padding: 1.2rem;
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.6) 0%, rgba(22, 33, 62, 0.6) 100%);
        border: 1px solid rgba(0, 255, 170, 0.3);
        border-radius: 6px;
        margin: 0 0.5rem;
        flex: 1;
        box-shadow: 0 2px 12px rgba(0, 255, 170, 0.1);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .stat-item:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(0, 255, 170, 0.2);
        border-color: rgba(0, 255, 170, 0.5);
    }
    
    .stat-item div:first-child {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 0 5px rgba(0, 255, 255, 0.3));
    }
    
    .stat-item div:nth-child(2) {
        font-family: 'Orbitron', monospace;
        font-weight: 600;
        font-size: 0.9rem;
        color: #00ffaa;
        margin-bottom: 0.3rem;
    }
    
    .stat-item div:last-child {
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.85rem;
        color: #a0a0a0;
    }
    
    .emoji-large {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        filter: drop-shadow(0 0 8px rgba(0, 255, 255, 0.4));
    }
    
    /* Streamlit specific overrides */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00ffaa 0%, #00d4ff 100%);
        box-shadow: 0 0 10px rgba(0, 255, 170, 0.3);
    }
    
    .stFileUploader > div {
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 6px;
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.4) 0%, rgba(22, 33, 62, 0.4) 100%);
        backdrop-filter: blur(10px);
    }
    
    .stMarkdown h3 {
        font-family: 'Orbitron', monospace;
        color: #00d4ff;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 1.1rem;
        text-shadow: 0 0 5px rgba(0, 212, 255, 0.3);
    }
    
    .stMarkdown h1, .stMarkdown h2 {
        font-family: 'Orbitron', monospace;
        color: #00d4ff;
    }
    
    .stExpanderHeader {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%) !important;
        border: 1px solid rgba(0, 255, 255, 0.3) !important;
        color: #00ffaa !important;
        font-family: 'Orbitron', monospace !important;
        font-size: 0.9rem !important;
        border-radius: 6px !important;
    }
    
    .stSpinner > div {
        border-top-color: #00ffaa !important;
    }
    
    .stAlert {
        background: linear-gradient(135deg, rgba(26, 26, 46, 0.8) 0%, rgba(22, 33, 62, 0.8) 100%) !important;
        border: 1px solid rgba(0, 255, 170, 0.4) !important;
        border-radius: 6px !important;
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="main-header">
    <div class="main-title">Neural Beast Detector</div>
    <div class="subtitle">Cyberpunk Vision AI • Cat vs Dog Classification</div>
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
        <h3>System Protocol</h3>
        <p>1. Upload target image (JPG, JPEG, PNG format)</p>
        <p>2. Neural network executes deep analysis via Vision Transformer</p>
        <p>3. Receive classification result with confidence metrics</p>
    </div>
    """, unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader(
    "🔍 Upload Target Image",
    type=["jpg", "jpeg", "png"],
    help="Select image file: JPG, JPEG, or PNG format"
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
                caption="📸 Target Image", 
                use_container_width=True,
                width=300
            )
        
        with result_col:
            st.success("✅ Neural Network Online")
            
            # Processing animation
            with st.spinner("⚡ Processing Neural Scan..."):
                # Preprocess and predict
                img_tensor = transform(image).unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(img_tensor)
                    probabilities = torch.softmax(output, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
            
            class_labels = ["Feline Unit", "Canine Unit"]
            class_emojis = ["🐱", "🐶"]
            confidence = probabilities[predicted_class].item()
            confidence_threshold = 0.70
            
            # Results display
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            
            if confidence < confidence_threshold:
                st.markdown(f"""
                <div class="emoji-large">❓</div>
                <div class="prediction-text uncertain-result">Unknown</div>
                <div class="confidence-text">
                    Classification inconclusive
                </div>
                <div style="margin-top: 1rem; font-size: 1rem;">
                    Confidence: {confidence:.1%}
                </div>
                """, unsafe_allow_html=True)
            else:
                result_class = "cat-result" if predicted_class == 0 else "dog-result"
                st.markdown(f"""
                <div class="emoji-large">{class_emojis[predicted_class]}</div>
                <div class="prediction-text {result_class}">{class_labels[predicted_class]}</div>
                <div class="confidence-text">Confidence: {confidence:.1%}</div>
                """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Progress bars for both classes
            st.markdown("### Analysis Report")
            
            cat_prob = probabilities[0].item()
            dog_prob = probabilities[1].item()
            
            st.markdown("🐱 **Feline Unit:**")
            st.progress(cat_prob)
            st.markdown(f"**{cat_prob:.1%}**")
            
            st.markdown("🐶 **Canine Unit:**")
            st.progress(dog_prob)
            st.markdown(f"**{dog_prob:.1%}**")
    
    except UnidentifiedImageError:
        st.error("❌ Invalid image file - corrupted or unsupported format")
    except Exception as e:
        st.error(f"❌ System error: {e}")

# Footer with stats
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-item">
        <div>🧠</div>
        <div>Neural Engine</div>
        <div>Vision Transformer</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-item">
        <div>⚡</div>
        <div>Real-time</div>
        <div>Instant Analysis</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-item">
        <div>🎯</div>
        <div>High Precision</div>
        <div>Accurate Results</div>
    </div>
    """, unsafe_allow_html=True)

# Additional info
with st.expander("⚙️ System Information"):
    st.markdown("""
    **Neural Beast Detector** menggunakan arsitektur Vision Transformer (ViT) untuk 
    klasifikasi gambar dengan akurasi tinggi. Sistem ini telah dioptimalkan untuk 
    mengenali perbedaan antara kucing dan anjing dengan presisi maksimal.
    
    **Technical Specifications:**
    - Core Architecture: ViT base patch16_224
    - Automatic image preprocessing
    - Confidence threshold untuk hasil akurat
    - Cyberpunk-themed interface
    - Real-time processing capability
    """)

st.markdown("""
<div style="text-align: center; margin-top: 2rem; opacity: 0.6; font-family: 'Rajdhani', sans-serif; font-size: 0.9rem;">
    Powered by Neural Network Technology
</div>
""", unsafe_allow_html=True)
