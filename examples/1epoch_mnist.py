import tensorflow as tf

import pyhopper
import numpy as np


def train_1ep_mnist(hparams):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)) / 255.0
    model = tf.keras.models.Sequential(
        [
            tf.keras.Input((28, 28, 1), dtype=tf.uint8),
            tf.keras.layers.Conv2D(16, 5, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, "relu"),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(10, "softmax"),
        ]
    )
    model.compile(
        loss=tf.keras.losses.sparse_categorical_crossentropy,
        metrics=[tf.keras.metrics.sparse_categorical_accuracy],
        optimizer=tf.keras.optimizers.Adam(0.0005),
    )
    hist = model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=8,
        validation_split=0.05,
    )
    return hist.history["val_sparse_categorical_accuracy"][-1]


search = pyhopper.Search(
    {
        "lr": pyhopper.float(1e-4, 1e-2),
    }
)
policy = search.run(
    pyhopper.wrap_n_times(train_1ep_mnist, n=5, yield_after=4),
    direction="max",
    seeding_steps=10,
    timeout="5s",
    n_jobs=4,
    kwargs={},
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(search.history.seconds, search.history.fs)
ax.plot(
    search.history.seconds,
    search.history.best_fs,
    label="Best",
    color="red",
)
fig.savefig("test.png")