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

The PyHopper's tuning process is a powerful **Markov chain Monte Carlo** (MCMC) sampler that

- runs parallel across multiple CPUs and GPUs
- natively supports NumPy array parameters with millions of dimensions
- is highly customizable (e.g. you can directly tune entire :code:`torch.Tensor` hyperparameters)

What's so special about PyHopper?
------------

There exist many powerful hyperparameter tuning frameworks out there. However, we created PyHopper because we weren't fully satisfied with any them.
In our experience they often showcase some fancy integration with other tools, at the cost of usability. PyHopper puts the user first.

- **You define when the results are ready, PyHopper takes care of the rest.** Different to many existing tuning tools that treat a timeout argument only as a termination signal, PyHopper carefully manages the exploration-exploitation tradeoff based on the provided timeout. You need your results by Monday? PyHopper will do exploration on Saturday and exploitation of Sunday.
- **A single algorithm is enough.** Many other tuning frameworks provide dozens of different tuning algorithms for the user to choose. While all these algorithm have their own use-cases where they excel, they also create a new meta-problem of deciding which algorithm is best suited for your task. So, instead of finding the optimal learning rate, we are now faced with the problem of finding the optimal tuning algorithm.
- **PyHopper is lightweight and hackable.** As an indirect consequence of the point above, other tuning frameworks often consist of thousands of lines of code with several layers of abstraction, making it difficult to implement custom types and sampling strategies. With PyHopper, you can implement custom parameter types and sampling strategies with just two lines of code.
- **Manual tuning naturally integrates with PyHopper.** For many ML problems, we often start with some manual hyperparameter tuning before running an automated tool. Moreover, form our experience we have some intuitions of what parameters could work. With PyHopper, you can guess some of the parameters. PyHopper then automatically fills the remaining parameters with the currently optimal values and evaluates your guess.
- **Why not Bayesian optimization?** While Bayesian optimization methods are quite effective for tuning lower dimensional hyperparameters (say less than 10), MCMC usually surpass them at higher dimensional problems.

The PyHopper's simple user interface allows running hyperparameter searches with less than 5 minutes setup time and minimal adaptations of existing code.

User’s Guide
------------

.. toctree::
    :maxdepth: 2

    quickstart
    walkthrough
    recipes

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
    :maxdepth: 2

    api