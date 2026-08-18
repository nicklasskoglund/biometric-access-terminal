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


def build_age_gender_model(
    input_dim: int = 512,
    num_age_classes: int = 101,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """Bygger ett delat nätverk för samtidig ålders- och könsestimering.

    Arkitektur enligt DEX-metoden (Rothe et al.), etablerad för just
    IMDB-WIKI-datasetet: ålder behandlas som klassificering över
    åldersbins (0-100 år) snarare än ren regression, där den slutgiltiga
    åldersprediktionen sedan beräknas som softmax-outputens förväntade
    värde av anroparen (notebooken/inferenskoden). Kön förblir binär
    klassificering.

    En gemensam trunk (input_dim -> 256 -> 128) lär sig en delad
    representation, som sedan grenar ut i två separata, mindre huvuden -
    ett per uppgift - så att varje huvud kan specialisera sig utan att
    dela exakt samma sista lager.

    Parameters
    ----------
    input_dim : int, default 512
        Dimensionen på ArcFace-embeddingen som matas in.
    num_age_classes : int, default 101
        Antal åldersbins (0-100 år, ett år per klass), enligt DEX.
    dropout_rate : float, default 0.3
        Andel noder som slumpmässigt nollställs per dropout-lager under
        träning.

    Returns
    -------
    keras.Model
        Okompilerad modell med två namngivna outputs ("age_output",
        "gender_output"). Kompileras separat av anroparen, som avgör
        optimizer/loss/loss_weights vid träning - se build_authorization_
        classifier() för samma designval och motivering.

    Notes
    -----
    "age_output" ska kompileras med categorical_crossentropy mot
    one-hot-encodade åldersklasser (avrundad ålder, clippad till
    [0, num_age_classes - 1]). "gender_output" ska kompileras med
    binary_crossentropy, som i build_authorization_classifier().
    """
    inputs = keras.Input(shape=(input_dim,))

    x = layers.Dense(256, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    age_branch = layers.Dense(64, activation="relu")(x)
    age_output = layers.Dense(
        num_age_classes, activation="softmax", name="age_output"
    )(age_branch)

    gender_branch = layers.Dense(32, activation="relu")(x)
    gender_output = layers.Dense(
        1, activation="sigmoid", name="gender_output"
    )(gender_branch)

    return keras.Model(
        inputs=inputs,
        outputs=[age_output, gender_output],
        name="age_gender_model",
    )