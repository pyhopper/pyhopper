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

    def my_objective(params: dict) -> float:
        model = build_model(params["hidden_size"],...)
        # .... train and evaluate the model
        return val_accuracy

    search = pyhopper.Search(
        units   = pyhopper.int(100,500),
        dropout = pyhopper.float(0,0.4,"0.1f"), # 1 decimal digit
        lr      = pyhopper.float(1e-5,1e-2,"0.1g"), # loguniform, 1 significant
        matrix  = pyhopper.float(-1,1,shape=(20,20)), # numpy array
        opt     = pyhopper.choice(["adam","rmsprop","sgd"]),
    )
    best_params = search.run(my_objective, "maximize", "8h", n_jobs="per-gpu")

PyHopper is a **scheduled Markov chain Monte Carlo** (sMCMC) sampler that

- runs parallel across multiple CPUs and GPUs
- natively supports NumPy array parameters with millions of dimensions
- automatically focuses its search space based on the remaining runtime

User’s Guide
--------------

.. toctree::

    quickstart
    walkthrough
    recipes/index
    examples/index
    api/index