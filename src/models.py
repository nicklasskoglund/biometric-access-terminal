"""Klassificeringsmodeller för Biometric Access Terminal.

Denna modul innehåller de tränade neurala nätverk som bygger på ArcFace-
embeddings: dels åtkomstklassificeraren (auktoriserad/ej auktoriserad),
dels (senare) ålder/kön-estimeringsmodellen. Dessa lager är projektets
egna deep learning-bidrag - ArcFace-embeddingen i sig är förtränad och
används oförändrad (se embeddings.py).
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_authorization_classifier(
    input_dim: int = 512,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Bygger ett dense feedforward-nätverk för binär åtkomstklassificering.

    Arkitektur: input_dim -> 128 -> 32 -> 1 (sigmoid), med batch
    normalization och dropout mellan de dolda lagren för att motverka
    överanpassning - relevant här eftersom den positiva klassen bygger på
    ett begränsat antal unika identiteter (en enskild person, augmenterad).

    Parameters
    ----------
    input_dim : int, default 512
        Dimensionen på ArcFace-embeddingen som matas in.
    dropout_rate : float, default 0.3
        Andel noder som slumpmässigt nollställs per dropout-lager under
        träning.

    Returns
    -------
    keras.Model
        Okompilerad modell. Kompileras separat av anroparen (notebooken),
        som även avgör optimizer/loss/class_weight vid träning.

    Notes
    -----
    Modellen returneras okompilerad med avsikt - kompilering (optimizer,
    loss, metrics) hör till träningskonfigurationen, inte arkitekturen,
    och hålls därför i notebooken där hyperparametrar som class_weight
    också sätts.
    """
    inputs = keras.Input(shape=(input_dim,))

    x = layers.Dense(128, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(32, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="authorization_classifier")