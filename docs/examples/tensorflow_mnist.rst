MLP with TensorFlow on MNIST
-------------------------------------

A typical use-case with Tensorflow would look something like this

.. code-block:: python

    import tensorflow as tf
    import pyhopper
    import numpy as np

    def get_data(for_validation=True):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape((-1, 28 * 28)) / 255.0
        x_test = x_test.reshape((-1, 28 * 28)) / 255.0
        if for_validation:
            val_size = int(0.05 * x_train.shape[0])
            shuffle = np.random.default_rng(12345).permutation(x_train.shape[0])
            x_train, y_train = x_train[shuffle], y_train[shuffle]
            x_val, y_val = x_train[:val_size], y_train[:val_size]
            x_train, y_train = x_train[val_size:], y_train[val_size:]
            return x_train, y_train, x_val, y_val
        else:
            return x_train, y_train, x_test, y_test


    def train_mnist_mlp(params, for_validation=True):
        x_train, y_train, x_val, y_val = get_data(for_validation)
        input_tensor = tf.keras.Input((28 * 28))
        x = input_tensor
        for i in range(params["num_layers"]):
            x = tf.keras.layers.Dense(
                params["size"][i],
                activation=params["activation"],
                kernel_regularizer=tf.keras.regularizers.l2(params["weight_decay"]),
            )(x)
            x = tf.keras.layers.Dropout(params["dropout"])(x)
        x = tf.keras.layers.Dense(10, "softmax")(x)
        model = tf.keras.Model(input_tensor, x)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            params["lr_init"],
            params["epochs"] * len(x_train) // params["batch_size"],
            params["alpha"],
        )
        model.compile(
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=[tf.keras.metrics.sparse_categorical_accuracy],
            optimizer=tf.keras.optimizers.Adam(lr_schedule),
        )
        model.fit(
            x_train,
            y_train,
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            validation_data=None if for_validation else (x_val, y_val),
            verbose=0,
        )
        _, val_acc = model.evaluate(x_val, y_val, verbose=0)
        return val_acc


    search = pyhopper.Search(
        {
            "activation": pyhopper.choice(["relu", "swish", "gelu", "elu"]),
            "num_layers": pyhopper.int(1, 5),
            "size": pyhopper.int(64, 256, multiple_of=16, shape=5),
            "dropout": pyhopper.float(0, 0.5, precision=1),
            "lr_init": pyhopper.float(0.005, 0.0005, log=True),
            "alpha": pyhopper.choice([0, 1e-3, 1e-2, 1 - 1], is_ordinal=True),
            "weight_decay": pyhopper.float(1e-6, 1e-2, log=True, precision=1),
            "batch_size": 32,
            "epochs": 30,
        }
    )
    best_params = search.run(
        pyhopper.wrap_n_times(train_mnist_mlp, n=3, yield_after=0),
        direction="max",
        timeout="4h",
        n_jobs="per-gpu",
        canceler=pyhopper.cancelers.QuantileCanceler(0.6),
    )
    test_acc = train_mnist_mlp(best_params, for_validation=False)
    print(f"Tuned params test accuracy: {100*test_acc:0.2f}%")
    print("best", best_params)

Outputs

.. code-block:: text

    > Search is scheduled for 04:00:00 (h:m:s)
    > Best f: 0.989 (out of 127 params):  98%|█████████▊| [3:56:08<03:49, 52.3 s/param]
    > ============================ Summary ===========================
    > Mode              : Best f : Steps : Canceled : Time
    > ----------------  : ----   : ----  : ----     : ----
    > Initial solution  : 0.983  : 1     : 0        : 09:48 (m:s)
    > Random seeding    : 0.986  : 26    : 48       : 04:39:53 (h:m:s)
    > Local sampling    : 0.989  : 100   : 96       : 16:03:12 (h:m:s)
    > ----------------  : ----   : ----  : ----     : ----
    > Total             : 0.989  : 127   : 144      : 03:56:10 (h:m:s)
    > ================================================================
    >
    > Tuned params test accuracy: 98.74%
    > best {'activation': 'swish', 'num_layers': 2, 'size': array([240, 208, 160, 128, 144]), 'dropout': 0.1, 'lr_init': 0.00239950593715168, 'alpha': 0.001, 'weight_decay': 2e-05, 'batch_size': 32, 'epochs': 30}


.. note::

    Achieving a >99.5% accuracy on MNIST is possible with a convolutional neural network