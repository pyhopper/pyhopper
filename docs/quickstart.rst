Quickstart
==========

This is the quickstart guide of PyHopper

The basics
--------------

The central component of PyHopper is the :meth:`pyhopper.Search` class. Its constructor requires a :meth:`dict` object that
acts as as the hyperparameter *template* and defines the search space.
During runtime, PyHopper samples candidates by instantiating concrete hyperparameter from this template.
The resulting candidates are :meth:`dict` objects as well, with the only difference that PyHopper template types are replaced by a sample obtained from the MCMC core. For example

.. code-block:: python

    import pyhopper

    def of(param):
        print(param)
        return 0

    search = pyhopper.Search(
        {
            "my_const": "cifar10",
            "my_int": pyhopper.int(100,500),
            "my_float": pyhopper.float(0,0.4),
            "my_choice": pyhopper.choice(["adam","rmsprop","sgd"]),
        }
    )
    search.run(of,max_steps=5)

outputs

.. code-block:: text

    >>> {'my_const': 'cifar10', 'my_int': 442, 'my_float': 0.16690259272502092, 'my_choice': 'sgd'}
    >>> {'my_const': 'cifar10', 'my_int': 198, 'my_float': 0.21107963087889003, 'my_choice': 'adam'}
    >>> {'my_const': 'cifar10', 'my_int': 159, 'my_float': 0.09813299118201196, 'my_choice': 'adam'}
    >>> {'my_const': 'cifar10', 'my_int': 203, 'my_float': 0.19852373670299772, 'my_choice': 'adam'}

Hyperparameter types
--------------

As shown above, PyHopper has three built-in template types: :meth:`int`, :meth:`float`, and :meth:`choice` (see :ref:`API docs<parameters>`).

:meth:`pyhopper.int` requires a lower and an upper bound (inclusive bounds) defining the range of the search space.
Optionally, we can provide an initial *guess* for the parameter via the `init` argument. Moreover, as Nvidia's TensorCores require the batch size
and other parameters to be a multiple of 8, we can constraint the parameter to multiples of some constant.

.. code-block:: python

    search = pyhopper.Search(
        {
            "layers": pyhopper.int(1, 8),
            "epochs": pyhopper.int(10, 50, init=30),
            "batch_size": pyhopper.int(32, 128, multiple_of=32),
        }
    )

generates samples

.. code-block:: text

    >>> {'layers': 2, 'epochs': 30, 'batch_size': 64}
    >>> {'layers': 3, 'epochs': 39, 'batch_size': 32}
    >>> {'layers': 8, 'epochs': 48, 'batch_size': 128}
    >>> {'layers': 3, 'epochs': 26, 'batch_size': 64}
    >>> {'layers': 5, 'epochs': 35, 'batch_size': 96}

:meth:`pyhopper.float`, similar to before, accepts inclusive lower and upper bounds and an optional initial guess.
Hyperparameters often span over multiple orders of magnitude. For instance, the optimal learning rate of a neural network
could be in the range from 0.00001 to 0.1.
Drawing uniform samples from this range favors larger values, as the center of the interval is approximately 0.05, which means that half of all generated samples will be larger than 0.05 on average.

For such parameters, **logarithmic** sampling, enabled via the :code:`log` argument, is a better option

.. code-block:: python

    search = pyhopper.Search(
        {
            "dropout": pyhopper.float(0, 0.5),
            "lr1": pyhopper.float(1e-5, 1e-1),           # uniform
            "lr2": pyhopper.float(1e-5, 1e-1, log=True), # logarithmic
        }
    )

.. code-block:: text

    >>> {"dropout": 0.1181678, "lr1": 0.0552744, "lr2": 0.0012332}
    >>> {"dropout": 0.0336810, "lr1": 0.0469721, "lr2": 0.0000148}
    >>> {"dropout": 0.1909593, "lr1": 0.0077057, "lr2": 0.0246946}
    >>> {"dropout": 0.1304118, "lr1": 0.0565307, "lr2": 0.0018307}
    >>> {"dropout": 0.2915319, "lr1": 0.0846803, "lr2": 0.0444826}

For most float parameters, keeping all digits is a) not necessary, b) looks ugly, and c) even makes the problem prone to overfitting.
To limit the precision of our float parameter, we can use the :code:`precision` argument.
In the default uniform sampling mode, this argument defines the number of digits after the comma.
In the logarithmic mode, a sampled value is rounded to the defined number of significant digits.

.. code-block:: python

    search = pyhopper.Search(
        {
            "dropout": pyhopper.float(0, 0.5, precision=2),
            "lr2": pyhopper.float(1e-5, 1e-1, log=True, precision=1),
        }
    )

.. code-block:: text

    >>> {'dropout': 0.04, 'lr2': 0.0001}
    >>> {'dropout': 0.11, 'lr2': 0.02}
    >>> {'dropout': 0.37, 'lr2': 0.008}
    >>> {'dropout': 0.13, 'lr2': 0.0001}
    >>> {'dropout': 0.2, 'lr2': 0.0009}


:meth:`pyhopper.choice` requires a :code:`list` of possible values for this hyperparameter.
Similar to before, we can provide an initial guess.
In case the values in the list are provided in a **structured order**, setting the :code:`is_ordinal` argument indicates pyhopper to preserve this order when sampling.
For instance, in the example below, the parameter :code:`"opt"` has no ordering but :code:`"dropout"` has, making pyhopper sample only adjacent items.

.. code-block:: python

    search = pyhopper.Search(
        {
            "opt": pyhopper.choice(["adam", "rmsprop", "sgd"]),
            "dropout": pyhopper.choice([0, 0.1, 0.2, 0.3], is_ordinal=True),
        }
    )

.. code-block:: text

    {'opt': 'adam', 'dropout': 0.3}
    {'opt': 'adam', 'dropout': 0.2}
    {'opt': 'sgd', 'dropout': 0.3}

Running PyHopper
--------------

Once we have defined the search space, we can schedule the search using the :meth:`pyhopper.Search.run` method.
The method requires three argument: The objective function, the direction of the search (minimize or maximize), and runtime of the search.

.. code-block:: python

    def my_objective_function(param: dict) -> float:
        return param["x"]

    search = pyhopper.Search(
        {
            "x": pyhopper.float(0,1),
        }
    )

    search.run(my_objective_function,"minimize","2s")

For specifying the runtime, we can provide a string, for instance :code:`"3d 7h 30m 10s"` is parsed to 3 days, 7 hours, 30 minute and 10 seconds, or simply an integer/float with the runtime in seconds.

To utilize multi CPU/GPU hardware more effectively, we can run multiple evaluations of parameter candidates in parallel with the :code:`n_jobs` argument.
For instance, the second call of :code:`run` in

.. code-block:: python

    from time import sleep

    def my_objective(param):
        sleep(1)
        return param["x"]

    search = pyhopper.Search({"x": pyhopper.float(0, 1)})

    search.run(my_objective, "minimize", "3s")
    search.run(my_objective, "minimize", "3s", n_jobs=4)

spawns 4 worker process resulting in much more evaluated candidates.

Setting the argument to :code:`n_jobs="per-gpu"` will spawn exactly one worker process for each GPU attached to the machine.
Moreover, PyHopper will take care of setting the :code:`CUDA_VISIBLE_DEVICES` environment variable for each of the worker processes to its private GPU, so each worker *sees* only a single GPU.
Consequently, we can write standard PyTorch and TensorFlow code in the objective function without having to worry about two processes accessing the same device.

Evaluating several initial weights
--------------

Training a neural network is an inherently stochastic process. Especially, the initial weights have a strong influence in the resulting accuracy.
As a result, during a hyperparameter search it may happen that a subpar parameter candidate was *lucky* with the specific initial weights used when training the network with the candidate parameters.
To tell spurious and genuine high accuracies apart we have to evaluate each parameter candidate several times and use the average accuracy as our objective metric.
For exactly this reason, PyHopper provides the :meth:`pyhopper.wrap_n_times` function that wraps an arbitrary function into its mean over n evaluations.

.. code-block:: python

    def my_objective(param):
        print(param["name"])
        return 0

    search = pyhopper.Search({"name": pyhopper.choice(["adam","eve"])})

    search.run(
        pyhopper.wrap_n_times(my_objective,3),
        "minimize",
        "3s"
    )

.. code-block:: text

    >>> adam
    >>> adam
    >>> adam
    >>> eve
    >>> eve
    >>> eve

A final Copy-Paste snippet
--------------

Putting everything together, a typical hyperparameter tuning code may look something like this

.. code-block:: python

    import pyhopper

    def my_objective(param: dict) -> float:
        # Add code here
        return val_acc

    search = pyhopper.Search(
        {
            "epochs": 20,
            "num_layers": pyhopper.int(1, 8, init=4),
            "batch_size": pyhopper.int(32, 512, multiple_of=32),
            "dropout": pyhopper.float(0, 0.5, precision=1),
            "lr": pyhopper.float(1e-5, 1e-2, log=True, precision=1),
            "opt": pyhopper.choice(["adam", "rmsprop", "sgd"], init="adam"),
            "weight_decay": pyhopper.choice([0, 1e-5, 1e-4, 1e-3], is_ordinal=True),
        }
    )
    search.run(pyhopper.wrap_n_times(my_objective,3), "max", "4h", n_jobs="per-gpu")

More advanced topics can be found in the rest of this documentation

.. toctree::
    :maxdepth: 2

    walkthrough
    recipes
    api