import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Configuration de la page
st.set_page_config(
    page_title="BrainScan AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
css = """
<style>
:root {
    --primary: #4361ee;
    --secondary: #3f37c9;
    --accent: #4cc9f0;
    --light: #f8f9fa;
    --dark: #212529;
    --success: #4caf50;
    --danger: #f94144;
    --warning: #f8961e;
}

body {
    font-family: 'Roboto', sans-serif;
    background-color: #f5f7fa;
    color: var(--dark);
    line-height: 1.6;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

.header-content {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: 3rem 2rem;
    border-radius: 12px;
    box-shadow: 0 10px 20px rgba(67, 97, 238, 0.15);
    text-align: center;
    margin-bottom: 3rem;
}

.header-content h1 {
    font-family: 'Montserrat', sans-serif;
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    max-width: 600px;
    margin: 0 auto;
}

.upload-section {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    margin-bottom: 2rem;
}

.upload-title {
    font-size: 1.3rem;
    margin-bottom: 1.5rem;
    color: var(--dark);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.upload-form {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

label {
    font-weight: 500;
    color: var(--dark);
}

.stButton > button {
    background: var(--primary);
    color: white;
    border: none;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
}

.stButton > button:hover {
    background: var(--secondary);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(67, 97, 238, 0.3);
}

.error {
    color: var(--danger);
    background: #ffebee;
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1.5rem;
    border-left: 4px solid var(--danger);
}

.result-card {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.05);
    margin-top: 2rem;
    animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #eee;
}

.result-title {
    font-size: 1.5rem;
    color: var(--dark);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.confidence {
    background: var(--primary);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 500;
    font-size: 0.9rem;
}

.image-container {
    display: flex;
    gap: 2rem;
    margin: 2rem 0;
}

.image-box {
    flex: 1;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
}

.image-box img {
    width: 100%;
    height: auto;
    display: block;
}

.image-label {
    background: var(--light);
    padding: 0.5rem;
    text-align: center;
    font-weight: 500;
}

.probs-container {
    margin-top: 1.5rem;
}

.prob-item {
    margin-bottom: 1rem;
}

.prob-label {
    display: flex;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}

.prob-name {
    font-weight: 500;
}

.prob-value {
    font-weight: 700;
    color: var(--primary);
}

.prob-bar {
    height: 10px;
    background: #eee;
    border-radius: 5px;
    overflow: hidden;
}

.prob-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--primary));
    border-radius: 5px;
    transition: width 0.5s ease;
}

footer {
    text-align: center;
    margin-top: 3rem;
    padding-top: 2rem;
    border-top: 1px solid #eee;
    color: #666;
    font-size: 0.9rem;
}

@media (max-width: 768px) {
    .container {
        padding: 1rem;
    }
    
    .header-content {
        padding: 2rem 1rem;
    }
    
    h1 {
        font-size: 1.8rem;
    }
    
    .image-container {
        flex-direction: column;
    }
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

# Noms des classes
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Prétraitement des images
IMG_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.5] * 3
NORMALIZATION_STD = [0.5] * 3

# Transformation PyTorch
pytorch_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
])

# Chargement du modèle PyTorch
@st.cache_resource
def load_pytorch_model():
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(CLASS_NAMES))
        )
        state_dict = torch.load("Cheikh_Fall_model.torch", map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except FileNotFoundError:
        st.error("Modèle PyTorch 'Cheikh_Fall_model.torch' introuvable. Vérifiez le chemin du fichier.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle PyTorch : {str(e)}")
        return None

# Chargement du modèle TensorFlow
@st.cache_resource
def load_tensorflow_model():
    try:
        model = tf.keras.models.load_model("Cheikh_Fall_model.tensorflow")
        return model
    except FileNotFoundError:
        st.error("Modèle TensorFlow 'Cheikh_Fall_model.tensorflow' introuvable. Vérifiez le chemin du fichier.")
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle TensorFlow : {str(e)}")
        return None

# Prétraitement pour PyTorch
def preprocess_pytorch(img):
    img = Image.open(img).convert('RGB')
    img_tensor = pytorch_transform(img).unsqueeze(0)
    
    # Image pour le débogage
    debug_img = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
    debug_img = debug_img * NORMALIZATION_STD + NORMALIZATION_MEAN
    debug_img = np.clip(debug_img, 0, 1)
    
    return img_tensor, debug_img

# Prétraitement pour TensorFlow
def preprocess_tensorflow(img):
    img = Image.open(img).convert('RGB')
    img = img.resize(IMG_SIZE)
    
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = (img_array / 127.5) - 1.0
    
    debug_img = (img_array[0].numpy() + 1.0) / 2.0
    
    return img_array, debug_img

# Fonction de prédiction
def predict(image, model_choice):
    if model_choice == "pytorch":
        model = load_pytorch_model()
        if model is None:
            return None
        img_tensor, debug_img = preprocess_pytorch(image)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            return {
                "class": CLASS_NAMES[preds.item()],
                "confidence": round(probs[0][preds.item()].item(), 4),
                "all_probs": {name: round(prob.item(), 4) for name, prob in zip(CLASS_NAMES, probs[0])},
                "debug_img": debug_img
            }
            
    elif model_choice == "tensorflow":
        model = load_tensorflow_model()
        if model is None:
            return None
        img_array, debug_img = preprocess_tensorflow(image)
        
        predictions = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(predictions[0])
        probs = predictions[0]  # Assume model output is already softmax; adjust if needed
        
        return {
            "class": CLASS_NAMES[pred_idx],
            "confidence": round(float(probs[pred_idx]), 4),
            "all_probs": {name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probs)},
            "debug_img": debug_img
        }

# Application principale
def main():
    # En-tête
    st.markdown("""
    <div class="header-content">
        <h1>BrainScan AI</h1>
        <p class="subtitle">Système avancé de classification des tumeurs cérébrales utilisant l'intelligence artificielle</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Section d'upload
    with st.container():
        st.markdown("""
        <div class="upload-section">
            <h2 class="upload-title">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" style="vertical-align: middle;">
                    <path d="M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z"/>
                </svg>
                Analyser une image IRM
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Sélectionner une image IRM :", 
            type=["jpg", "jpeg", "png"], 
            key="file_uploader",
            label_visibility="collapsed"
        )
        
        model_choice = st.selectbox(
            "Choix du modèle :", 
            ["pytorch", "tensorflow"], 
            index=0,
            key="model_select"
        )
        
        if st.button("Lancer l'analyse", key="analyze_button"):
            if uploaded_file is not None:
                try:
                    # Affichage de l'image originale
                    img = Image.open(uploaded_file)
                    
                    # Prédiction
                    result = predict(uploaded_file, model_choice)
                    if result is None:
                        return
                    
                    # Affichage des résultats
                    with st.container():
                        st.markdown(f"""
                        <div class="result-card">
                            <div class="result-header">
                                <h2 class="result-title">
                                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="24" height="24" style="vertical-align: middle;">
                                        <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
                                    </svg>
                                    Résultat de l'analyse
                                </h2>
                                <div class="confidence">{round(result["confidence"] * 100, 1)}% de confiance</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Comparaison des images
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown('<div class="image-label">Image originale</div>', unsafe_allow_html=True)
                            st.image(img, use_column_width=True)
                        
                        with col2:
                            st.markdown('<div class="image-label">Image prétraitée</div>', unsafe_allow_html=True)
                            st.image(result["debug_img"], use_column_width=True, channels="RGB")
                        
                        # Probabilités
                        st.markdown("""
                        <div class="probs-container">
                            <h3>Probabilités par type de tumeur :</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for name, prob in result["all_probs"].items():
                            st.markdown(f"""
                            <div class="prob-item">
                                <div class="prob-label">
                                    <span class="prob-name">{name.capitalize()}</span>
                                    <span class="prob-value">{round(prob * 100, 1)}%</span>
                                </div>
                                <div class="prob-bar">
                                    <div class="prob-fill" style="width: {prob * 100}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"Erreur lors du traitement : {str(e)}")
            else:
                st.error("Veuillez sélectionner une image avant de lancer l'analyse.")
    
    # Pied de page
    st.markdown("""
    <footer>
        <p>BrainScan AI - Système de diagnostic assisté par IA © 2025</p>
        <p>Pour usage médical professionnel uniquement</p>
    </footer>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()