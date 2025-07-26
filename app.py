import streamlit as st
import torch
import timm
import torch.nn as nn
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download

# Judul halaman
st.set_page_config(page_title="Klasifikasi Kucing vs Anjing", layout="centered")
st.title("🐱🐶 Klasifikasi Gambar Kucing vs Anjing dengan ViT")

# Deteksi perangkat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformasi gambar
ukuran_gambar = 224
transform = transforms.Compose([
    transforms.Resize((ukuran_gambar, ukuran_gambar)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Fungsi untuk memuat model
@st.cache_resource
def muat_model():
    model = timm.create_model("vit_base_patch16_224", pretrained=False)
    model.head = nn.Linear(model.head.in_features, 2)
    model.load_state_dict(torch.load("vit_model.pth", map_location=device))  # Pastikan file model tersedia
    model.to(device)
    model.eval()
    return model

model = muat_model()

# Upload gambar
uploaded_file = st.file_uploader("Unggah gambar kucing atau anjing", type=["jpg", "jpeg", "png"])

# Logika prediksi
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

        st.markdown("### Hasil Prediksi")

        # Pra-pemrosesan dan prediksi
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            probabilitas = torch.softmax(output, dim=1)[0]
            kelas_terprediksi = torch.argmax(probabilitas).item()

        label_kelas = ["Kucing", "Anjing"]
        keyakinan = probabilitas[kelas_terprediksi].item()

        # Threshold keyakinan
        ambang_keyakinan = 0.90

        if keyakinan < ambang_keyakinan:
            st.markdown(
                f"<div style='font-size: 20px; font-weight: bold; color: orange;'>Kelas: Bukan Kucing atau Anjing</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 16px;'>Model tidak cukup yakin untuk mengklasifikasikan gambar ini sebagai kucing atau anjing (Keyakinan: {keyakinan:.2%}).</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"<div style='font-size: 20px; font-weight: bold;'>Kelas: {label_kelas[kelas_terprediksi]}</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='font-size: 16px;'>Tingkat Keyakinan: {keyakinan:.2%}</div>",
                unsafe_allow_html=True
            )

    except UnidentifiedImageError:
        st.error("File yang diunggah bukan gambar yang valid atau rusak.")
    except Exception as e:
        st.error(f"Terjadi kesalahan yang tidak terduga: {e}")
