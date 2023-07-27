import numpy as np
from tensorflow import keras
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops

from constants import *


# https://stackoverflow.com/questions/48485870/multi-label-classification-with-class-weights-in-keras
def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        y_true = math_ops.cast(y_true, y_pred.dtype)
        return K.mean(
            (weights[:, 0] ** (1 - y_true)) * (weights[:, 1] ** (y_true)) * K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss


def get_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', classes=np.unique(y_true[:, i]), y=y_true[:, i])
    return weights


def nn_predictor(X_train, y_train):
    input_shape = X_train.shape[1]
    output_units = y_train.shape[1]
    layers = [
        keras.layers.Input(shape=input_shape),
        keras.layers.Flatten()]
    for n in range(0, HIDDEN_LAYERS):
        layers.append(keras.layers.Dense(units=LAYER_WIDTH, activation="relu"))
        layers.append(keras.layers.Dropout(DROPOUT))
    layers.append(keras.layers.Dense(units=output_units, activation="sigmoid"))

    model = keras.Sequential(layers)
    smell_weights = get_class_weights(y_train)
    loss = get_weighted_loss(smell_weights)

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
    )
    return model
