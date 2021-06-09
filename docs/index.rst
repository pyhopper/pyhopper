.. PyHopper documentation master file, created by
sphinx-quickstart on Tue Mar 16 13:09:22 2021.
You can adapt this file completely to your liking, but it should at least
contain the root `toctree` directive.

Welcome to PyHopper's documentation!
====================================


PyHopper is a python hyperparameter optimizer, installable with

.. code-block:: bash

    pip3 install -U pyhopper

, straightforward to use

.. code-block:: python

    def objective(params: dict) -> float:
        model = build_model(params["hidden_size"],params["lr"],params["opt"])
        # .... train and evaluate the model
        return val_accuracy

    search = pyhopper.Search(
        {
            "hidden_size": pyhopper.int(100,500),
            "lr": pyhopper.float(0.00001,0.1,log=True), # logarithmic sampling
            "opt": pyhopper.choice(["adam","rmsprop","sgd"],init="adam"),
        }
    )
    best_params = search.run(objective, "max", "1h 30min")

and rich in useful features

.. code-block:: python

    best_params = search.run(
        pyhopper.wrap_n_times(objective,n=5,reduction="mean"), # average over 5 initializations
        "max",
        n_jobs="per-gpu", # Train multiple parameters candidates in parallel
        runtime="12h",
        callbacks=pyhopper.callbacks.FancyCallbackHERE(),
    )


User’s Guide
------------

Here is a list of recipes covering 99% of PyHopper's typical use-cases

.. toctree::
    :maxdepth: 2

    quickstart
    recipes

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
    :maxdepth: 2

    api