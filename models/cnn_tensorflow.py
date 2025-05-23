import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

def get_pretrained_model():
    # Charger ResNet50 préentraîné sans la dernière couche de classification
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Geler toutes les couches
    base_model.trainable = False

    # Construire la nouvelle tête de classification
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(4, activation='softmax')  
    ])

    return model
