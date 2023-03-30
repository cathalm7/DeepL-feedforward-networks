"""
The main code for the feedforward networks assignment.
See README.md for details.
"""
import os
from typing import Tuple, Dict
import tensorflow as tf

os.environ["TF_DETERMINISTIC_OPS"] = "1"
tf.random.set_seed(42)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

def create_auto_mpg_deep_and_wide_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tf.keras.models.Model,
                                                tf.keras.models.Model]:
    """Creates one deep neural network and one wide neural network.
    The networks should have the same (or very close to the same) number of
    parameters and the same activation functions.

    The neural networks will be asked to predict the number of miles per gallon
    that different cars get. They will be trained and tested on the Auto MPG
    dataset from:
    https://archive.ics.uci.edu/ml/datasets/auto+mpg

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (deep neural network, wide neural network)
    """
    # Deep neural network
    model_d = tf.keras.Sequential()
    model_d.add(tf.keras.layers.Dense(6, activation="relu", input_dim=n_inputs))
    model_d.add(tf.keras.layers.Dense(12, activation="relu"))
    model_d.add(tf.keras.layers.Dense(16, activation="relu"))
    model_d.add(tf.keras.layers.Dense(20, activation="relu"))
    model_d.add(tf.keras.layers.Dense(24, activation="relu"))
    model_d.add(tf.keras.layers.Dense(28, activation="relu"))
    model_d.add(tf.keras.layers.Dense(32, activation="relu"))
    model_d.add(tf.keras.layers.Dense(36, activation="relu"))
    model_d.add(tf.keras.layers.Dense(37, activation="relu"))
    model_d.add(tf.keras.layers.Dense(38, activation="relu"))
    model_d.add(tf.keras.layers.Dense(40, activation="relu"))
    model_d.add(tf.keras.layers.Dense(n_outputs, activation="linear"))
    model_d.compile(optimizer='rmsprop', loss='mse')

    # Wide neural network
    model_w = tf.keras.Sequential()
    model_w.add(tf.keras.layers.Dense(6, activation="relu", input_dim=n_inputs))
    model_w.add(tf.keras.layers.Dense(64, activation="relu"))
    model_w.add(tf.keras.layers.Dense(110, activation="relu"))
    model_w.add(tf.keras.layers.Dense(n_outputs, activation="linear"))
    model_w.compile(optimizer='rmsprop', loss='mse')

    return (model_d, model_w)

def create_delicious_relu_vs_tanh_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tf.keras.models.Model,
                                                tf.keras.models.Model]:
    """Creates one neural network where all hidden layers have ReLU activations,
    and one where all hidden layers have tanh activations. The networks should
    be identical other than the difference in activation functions.

    The neural networks will be asked to predict the 0 or more tags associated
    with a del.icio.us bookmark. They will be trained and tested on the
    del.icio.us dataset from:
    https://github.com/dhruvramani/Multilabel-Classification-Datasets
    which is a slightly simplified version of:
    https://archive.ics.uci.edu/ml/datasets/DeliciousMIL%3A+A+Data+Set+for+Multi-Label+Multi-Instance+Learning+with+Instance+Labels

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (ReLU neural network, tanh neural network)
    """

    # ReLU model
    model_relu = tf.keras.Sequential()
    model_relu.add(tf.keras.layers.Dense(16, activation="relu", input_dim=n_inputs))
    # add dropout layer to prevent overfitting
    model_relu.add(tf.keras.layers.Dropout(0.3))
    model_relu.add(tf.keras.layers.Dense(64, activation="relu"))
    model_relu.add(tf.keras.layers.Dense(128, activation="relu"))
    model_relu.add(tf.keras.layers.Dropout(0.5))
    model_relu.add(tf.keras.layers.Dense(n_outputs, activation="sigmoid"))

    #  BinaryFocalCrossentropy is chosen as it is a suitable choice for binary classification tasks
    model_relu.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # tanh model
    model_tanh = tf.keras.Sequential()
    model_tanh.add(tf.keras.layers.Dense(16, activation="tanh", input_dim=n_inputs))
    # add dropout layer to prevent overfitting
    model_tanh.add(tf.keras.layers.Dropout(0.3))
    model_tanh.add(tf.keras.layers.Dense(64, activation="tanh"))
    model_tanh.add(tf.keras.layers.Dense(128, activation="tanh"))
    model_tanh.add(tf.keras.layers.Dropout(0.5))
    model_tanh.add(tf.keras.layers.Dense(n_outputs, activation="sigmoid"))
    model_tanh.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return (model_relu, model_tanh)

def create_activity_dropout_and_nodropout_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tf.keras.models.Model,
                                                tf.keras.models.Model]:
    """Creates one neural network with dropout applied after each layer, and
    one neural network without dropout. The networks should be identical other
    than the presence or absence of dropout.

    The neural networks will be asked to predict which one of six activity types
    a smartphone user was performing. They will be trained and tested on the
    UCI-HAR dataset from:
    https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (dropout neural network, no-dropout neural network)
    """
    # Neural network with Drop Out
    model_drop = tf.keras.Sequential()
    model_drop.add(tf.keras.layers.Dense(6, activation="tanh", input_dim=n_inputs))
    # add dropout layer to prevent overfitting
    model_drop.add(tf.keras.layers.Dropout(0.1))
    model_drop.add(tf.keras.layers.Dense(32, activation="tanh"))
    model_drop.add(tf.keras.layers.Dense(64, activation="tanh"))
    model_drop.add(tf.keras.layers.Dropout(0.3))
    model_drop.add(tf.keras.layers.Dense(128, activation="tanh"))
    model_drop.add(tf.keras.layers.Dropout(0.5))
    model_drop.add(tf.keras.layers.Dense(n_outputs, activation="softmax"))
    # RMSprop adapts the learning rate based on the gradient history
    # Categorical_crossentropy penalizes incorrect predictions more strongly than correct predictions
    model_drop.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    # Neural network without Drop Out
    model_no_drop = tf.keras.Sequential()
    model_no_drop.add(tf.keras.layers.Dense(6, activation="tanh", input_dim=n_inputs))
    model_no_drop.add(tf.keras.layers.Dense(32, activation="tanh"))
    model_no_drop.add(tf.keras.layers.Dense(64, activation="tanh"))
    model_no_drop.add(tf.keras.layers.Dense(128, activation="tanh"))
    model_no_drop.add(tf.keras.layers.Dense(n_outputs, activation="softmax"))
    model_no_drop.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    return (model_drop, model_no_drop)

def create_income_earlystopping_and_noearlystopping_networks(
        n_inputs: int, n_outputs: int) -> Tuple[tf.keras.models.Model,
                                                Dict,
                                                tf.keras.models.Model,
                                                Dict]:
    """Creates one neural network that uses early stopping during training, and
    one that does not. The networks should be identical other than the presence
    or absence of early stopping.

    The neural networks will be asked to predict whether a person makes more
    than $50K per year. They will be trained and tested on the "adult" dataset
    from:
    https://archive.ics.uci.edu/ml/datasets/adult

    :param n_inputs: The number of inputs to the models.
    :param n_outputs: The number of outputs from the models.
    :return: A tuple of (
        early-stopping neural network,
        early-stopping parameters that should be passed to Model.fit,
        no-early-stopping neural network,
        no-early-stopping parameters that should be passed to Model.fit
    )
    """

    model_es = tf.keras.Sequential()
    model_es.add(tf.keras.layers.Dense(64, activation="tanh", input_dim=n_inputs))
    model_es.add(tf.keras.layers.Dense(64, activation="tanh"))
    model_es.add(tf.keras.layers.Dense(128, activation="tanh"))
    model_es.add(tf.keras.layers.Dense(256, activation="tanh"))
    model_es.add(tf.keras.layers.Dense(n_outputs, activation="sigmoid"))
    #  Stochastic Gradient Descent (SGD) algorithm as the optimization algorithm
    #  BinaryFocalCrossentropy is chosen as it is a suitable choice for binary classification tasks
    model_es.compile(optimizer='SGD', loss='BinaryFocalCrossentropy')

    # Validation loss will be monitored for early stopping
    # mode = Auto: Direction of the monitored quantity will be automatically inferred
    # Training will be stopped if there is no improvement in validation loss for 10 epochs
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=10)
    kwargs = {"callbacks":[earlystop]}


    model_noes = tf.keras.Sequential()
    model_noes.add(tf.keras.layers.Dense(64, activation="tanh", input_dim=n_inputs))
    model_noes.add(tf.keras.layers.Dense(64, activation="tanh"))
    model_noes.add(tf.keras.layers.Dense(128, activation="tanh"))
    model_noes.add(tf.keras.layers.Dense(256, activation="tanh"))
    model_noes.add(tf.keras.layers.Dense(n_outputs, activation="sigmoid"))
    model_noes.compile(optimizer='SGD', loss='BinaryFocalCrossentropy')

    return (model_es, kwargs, model_noes, {})
