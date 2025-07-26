import streamlit as st
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download

# Set title
st.set_page_config(page_title="Cat vs Dog Classifier", layout="centered")
st.title("🐱🐶 Cat vs Dog Classifier using ViT")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi gambar
image_size = 224
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load model
@st.cache_resource
def load_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load("vit_model.pth", map_location=device))  # Pastikan file ada di folder sama
    model.to(device)
    model.eval()
    return model

model = load_model()

# Upload gambar
uploaded_file = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg", "png"])

# Prediction logic
if uploaded_file is not None:
    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.markdown("### Prediction")

        # Preprocess and predict
        img_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            predicted_class = torch.argmax(probabilities).item()

        class_labels = ["Cat", "Dog"]
        confidence = probabilities[predicted_class].item()

        # Confidence threshold to determine if image might be neither
        confidence_threshold = 0.75

        if confidence < confidence_threshold:
            st.markdown(
                f"<div style='font-size: 20px; font-weight: bold; color: orange;'>Class: Neither Cat nor Dog</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 16px;'>The model is not confident enough to classify this image as a cat or a dog (Confidence: {confidence:.2%}).</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='font-size: 20px; font-weight: bold;'>Class: {class_labels[predicted_class]}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 16px;'>Confidence: {confidence:.2%}</div>",
                unsafe_allow_html=True
            )

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image or is corrupted.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
