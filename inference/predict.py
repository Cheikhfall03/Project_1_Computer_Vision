from torchvision import datasets, transforms, models
from PIL import Image
import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
import os

# Récupérer les classes depuis les dossiers
data_dir = 'data/training'
class_names = datasets.ImageFolder(data_dir).classes

# --------------------------
# ⚠️ Recréer exactement la même structure que celle entraînée
# --------------------------
model_pytorch = models.resnet50(pretrained=False)
num_ftrs = model_pytorch.fc.in_features
model_pytorch.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, len(class_names))
)

# Charger les poids
state_dict = torch.load("Cheikh_Fall_model.torch", map_location='cpu')
model_pytorch.load_state_dict(state_dict)
model_pytorch.eval()

# --------------------------
# Charger le modèle TensorFlow
# --------------------------
model_tf = tf.keras.models.load_model("Cheikh_Fall_model.tensorflow")

# --------------------------
# Prétraitement image
# --------------------------
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# --------------------------
# Fonction de prédiction
# --------------------------
def predict_class(image_path, model_choice):
    if model_choice == "pytorch":
        img_tensor = preprocess_image(image_path)
        with torch.no_grad():
            output = model_pytorch(img_tensor)
            _, predicted = torch.max(output, 1)
            return f"Class: {class_names[predicted.item()]}"
    elif model_choice == "tensorflow":
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model_tf.predict(img_array)
        return f"Class: {class_names[np.argmax(predictions)]}"
    else:
        return "Erreur : backend inconnu"
