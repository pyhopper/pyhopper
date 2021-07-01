:hide-toc:

=========================
PyHopper's documentation!
=========================

PyHopper is a black-box optimizer, made specifically for high-dimensional hyperparameter optimization problems arising in machine learning research.
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

The PyHopper's tuning algorithm is a powerful **Markov chain Monte Carlo** (MCMC) sampler that

- runs parallel across multiple CPUs and GPUs
- natively supports NumPy array parameters with millions of dimensions
- is highly customizable (e.g. you can directly tune entire :code:`torch.Tensor` hyperparameters)

User’s Guide
--------------

.. toctree::

    quickstart
    walkthrough
    recipes/index
    api/index