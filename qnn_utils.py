import time

import pennylane as qml
import tensorflow as tf
from pennylane import numpy as np
from pennylane.templates import RandomLayers
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm.notebook import tqdm

np.random.seed(0)
tf.random.set_seed(255)


def get_mnist_data(n_train=None, n_test=None):
    # load mnist
    mnist_dataset = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

    # restrict data
    if n_train:
        train_images, train_labels = train_images[:n_train], train_labels[:n_train]
    if n_test:
        test_images, test_labels = test_images[:n_test], test_labels[:n_test]

    # split test val
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2
    )

    # data normalization
    train_images = train_images / 255
    val_images = val_images / 255
    test_images = test_images / 255

    # add one more dimension
    train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
    val_images = np.array(val_images[..., tf.newaxis], requires_grad=False)
    test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

    return train_images, val_images, test_images, train_labels, val_labels, test_labels


def get_quantum_data(train_images, val_images, test_images, n_layers=1, n_wires=1, return_test=True):
    start_time = time.time()
    # init device
    dev = qml.device("default.qubit", wires=n_wires)
    # generate random parameters for quantum layers
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, n_wires))

    @qml.qnode(dev)
    def circuit(phi, n_wires):
        # Кодирование 3 классических входных данных
        for j in range(n_wires):
            qml.RY(np.pi * phi[j], wires=j)

        # Случайная квантовая цепь
        RandomLayers(rand_params, wires=list(range(n_wires)))

        # Измерения, которые дают 3 классических выходных значений для следующих слоев
        return [qml.expval(qml.PauliZ(j)) for j in range(n_wires)]

    def quanv(image, n_wires):
        """Функция квантовой свертки над входным изображением."""
        image = np.pad(image, ((1, 1), (1, 1), (0, 0)))
        out = np.zeros((28, 28, n_wires))

        # Циклы по координатам верхнего левого пикселя блоков 2х2
        for j in range(0, 28, 1):
            for k in range(0, 28, 1):
                # Обработка блока 2x2 из изображения квантовой цепью
                q_results = circuit(
                    [
                        image[j, k, 0],
                        image[j, k + 1, 0],
                        image[j + 1, k, 0],
                        image[j + 1, k + 1, 0],
                    ],
                    n_wires,
                )
                # Запись результатов наблюдения в выходной пиксель (j/2, k/2)
                for c in range(n_wires):
                    out[j, k, c] = q_results[c]
        return out

    # generate train images
    q_train_images = []
    for idx, img in enumerate(tqdm(train_images)):
        q_train_images.append(quanv(img, n_wires))
    q_train_images = np.asarray(q_train_images)

    # generate val images
    q_val_images = []
    for idx, img in enumerate(tqdm(val_images)):
        q_val_images.append(quanv(img, n_wires))
    q_val_images = np.asarray(q_val_images)

    if return_test:
        # generate test images
        q_test_images = []
        for idx, img in enumerate(tqdm(test_images)):
            q_test_images.append(quanv(img, n_wires))
        q_test_images = np.asarray(q_test_images)

        return time.time() - start_time, q_train_images, q_val_images, q_test_images
    return time.time() - start_time, q_train_images, q_val_images


def CNNModel():
    """Initializes and returns a custom Keras model
    which is ready to be trained."""
    model = keras.models.Sequential(
        [
            keras.layers.Conv2D(50, 5, activation="relu"),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64, 5, activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
