from flask import Flask, render_template, request, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import tensorflow as tf
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['DEBUG_FOLDER'] = 'static/debug'

# Configuration du logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Vérification et création des dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEBUG_FOLDER'], exist_ok=True)

# Noms des classes (doivent correspondre exactement à l'ordre utilisé pendant l'entraînement)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Configuration du prétraitement (doit être identique à celui utilisé pendant l'entraînement)
IMG_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.5] * 3  # Mean=0.5 comme pendant l'entraînement
NORMALIZATION_STD = [0.5] * 3   # Std=0.5 comme pendant l'entraînement

# Transformation PyTorch (identique à l'entraînement)
pytorch_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.Grayscale(num_output_channels=3),  # Conversion en 3 canaux
    transforms.ToTensor(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD)
])

def load_pytorch_model():
    """Charge le modèle PyTorch avec la même architecture que pendant l'entraînement"""
    try:
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, len(CLASS_NAMES))
        )
        
        # Charge les poids sauvegardés
        state_dict = torch.load("Cheikh_Fall_model.torch", map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        logger.info("Modèle PyTorch chargé avec succès")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle PyTorch: {str(e)}")
        raise

def load_tensorflow_model():
    """Charge le modèle TensorFlow sauvegardé"""
    try:
        model = tf.keras.models.load_model("Cheikh_Fall_model.tensorflow")
        logger.info("Modèle TensorFlow chargé avec succès")
        return model
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle TensorFlow: {str(e)}")
        raise

# Initialisation des modèles
model_pytorch = load_pytorch_model()
model_tf = load_tensorflow_model()

def save_debug_image(image_array, filename):
    """Sauvegarde une image pour le débogage"""
    try:
        plt.imshow(image_array)
        debug_path = os.path.join(app.config['DEBUG_FOLDER'], filename)
        plt.savefig(debug_path)
        plt.close()
        logger.info(f"Image de debug sauvegardée: {debug_path}")
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde de l'image de debug: {str(e)}")

def preprocess_pytorch(img_path):
    """Prétraitement pour PyTorch (identique à l'entraînement)"""
    try:
        # Ouverture et conversion de l'image
        img = Image.open(img_path).convert('RGB')
        
        # Application des transformations
        img_tensor = pytorch_transform(img).unsqueeze(0)
        
        # Sauvegarde de l'image prétraitée pour vérification
        debug_img = img_tensor.squeeze(0).permute(1, 2, 0).numpy()
        debug_img = debug_img * NORMALIZATION_STD + NORMALIZATION_MEAN  # Dénormalisation
        debug_img = np.clip(debug_img, 0, 1)
        save_debug_image(debug_img, 'debug_pytorch.png')
        
        return img_tensor
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement PyTorch: {str(e)}")
        raise

def preprocess_tensorflow(img_path):
    """Prétraitement pour TensorFlow (identique à l'entraînement)"""
    try:
        # Conversion en grayscale puis en RGB (3 canaux)
        img = Image.open(img_path).convert('L').convert('RGB')
        img = img.resize(IMG_SIZE)
        
        # Conversion en array et normalisation
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = (img_array / 127.5) - 1.0  # Équivalent à mean=0.5, std=0.5
        
        # Sauvegarde de l'image prétraitée pour vérification
        debug_img = (img_array[0].numpy() + 1) / 2.0  # Dénormalisation [0,1]
        save_debug_image(debug_img, 'debug_tensorflow.png')
        
        return img_array
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement TensorFlow: {str(e)}")
        raise

def predict(image_path, model_choice):
    """Effectue une prédiction avec le modèle spécifié"""
    try:
        if model_choice == "pytorch":
            img_tensor = preprocess_pytorch(image_path)
            
            with torch.no_grad():
                outputs = model_pytorch(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                logger.info(f"Sortie brute PyTorch: {outputs.numpy()}")
                logger.info(f"Probabilités PyTorch: {probs.numpy()}")
                
                return {
                    "class": CLASS_NAMES[preds.item()],
                    "confidence": round(probs[0][preds.item()].item(), 4),
                    "all_probs": {name: round(prob.item(), 4) for name, prob in zip(CLASS_NAMES, probs[0])},
                    "debug_img": url_for('static', filename='debug/debug_pytorch.png')
                }
                
        elif model_choice == "tensorflow":
            img_array = preprocess_tensorflow(image_path)
            
            predictions = model_tf.predict(img_array, verbose=0)
            pred_idx = np.argmax(predictions[0])
            probs = tf.nn.softmax(predictions[0]).numpy()
            
            logger.info(f"Sortie brute TensorFlow: {predictions}")
            logger.info(f"Probabilités TensorFlow: {probs}")
            
            return {
                "class": CLASS_NAMES[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
                "all_probs": {name: round(float(prob), 4) for name, prob in zip(CLASS_NAMES, probs)},
                "debug_img": url_for('static', filename='debug/debug_tensorflow.png')
            }
            
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {str(e)}", exc_info=True)
        return {"error": str(e)}

@app.route('/', methods=['GET', 'POST'])
def index():
    """Gère les requêtes pour la page principale"""
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', error="Aucune image fournie")
            
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', error="Aucun fichier sélectionné")
            
        try:
            # Sauvegarde du fichier uploadé
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Prédiction avec le modèle sélectionné
            model_choice = request.form.get('model', 'pytorch')
            result = predict(filepath, model_choice)
            
            if 'error' in result:
                return render_template('index.html', error=result['error'])
            
            return render_template('index.html',
                                prediction=result['class'],
                                confidence=result['confidence'],
                                all_probs=result['all_probs'],
                                image_url=url_for('static', filename=f'uploads/{filename}'),
                                debug_img=result.get('debug_img'))
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement: {str(e)}", exc_info=True)
            return render_template('index.html', error=f"Erreur lors du traitement: {str(e)}")
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')