import tensorflow as tf

def get_data():
    # Chargement des données d'entraînement
    train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "data/training/",
        labels='inferred',
        label_mode='int',
        image_size=(224, 224),
        batch_size=64,
        shuffle=True
    )

    # Chargement des données de test
    test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "data/testing/",
        labels='inferred',
        label_mode='int',
        image_size=(224, 224),
        batch_size=64,
        shuffle=False
    )

    # Normalisation (entre -1 et 1 pour matcher [-0.5, 0.5] en PyTorch)
    normalization_layer = tf.keras.layers.Rescaling(1./127.5, offset=-1)

    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    return train_dataset, test_dataset
