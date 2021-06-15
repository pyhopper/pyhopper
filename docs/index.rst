Welcome to PyHopper's documentation!
====================================


PyHopper is a hyperparameter optimizer, made specifically for high-dimensional problems arising in deep learning.
Quick to install

.. code-block:: bash

    pip3 install -U pyhopper

and straightforward to use

.. code-block:: python

    import pyhopper

    def objective(params: dict) -> float:
        model = build_model(params["hidden_size"],...)
        # .... train and evaluate the model
        return val_accuracy

    search = pyhopper.Search(
        {
            "hidden_size": pyhopper.int(100,500),
            "dropout_rate": pyhopper.float(0,0.4),
            "opt": pyhopper.choice(["adam","rmsprop","sgd"]),
        }
    )
    best_params = search.run(objective, "maximize", "1h 30min")

The PyHopper's search process is a powerful evolutionary strategy that
runs parallel across multiple GPUs
natively supports NumPy array parameters with millions of dimensions
is highly customizable (e.g. directly tuning torch.Tensor parameters)

The PyHopper's simple user interface allows running hyperparameter searches with less than 5 minutes setup time and minimal adaptations of existing code.

User’s Guide
------------

.. toctree::
    :maxdepth: 2

    quickstart
    search
    recipes

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
    :maxdepth: 2

    api