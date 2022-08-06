.. _pruning-label:

Discontinuing unpromising candidates during evaluation
------------------------------------------------------------------------

As discussed in the quickstart guide, the training process of a neural network is stochastic, making single evaluations of a hyperparameter candidate not fully trustworthy.
To avoid getting fooled by outliers, PyHopper provides the :meth:`pyhopper.wrap_n_times` wrapper function that evaluates a function multiple times and returns its mean.
However, evaluating every candidate several times comes with quite a high computational cost.
Therefore, **it would be great if there was a feature in Python for a function to return intermediate results**, for PyHopper to then discontinue the evaluation of unpromising candidates.
Luckily, Python's `generators functions <https://docs.python.org/3/reference/expressions.html#yield-expressions>`_ provide exactly that.

In particular, PyHopper supports Python generators functions as objective function that do not return a single float value but iterate over a sequence of floats.

.. code-block:: python

    def dummy_of(param):
        yield 0  # a quick estimate of the objective value that can be used for pruning the evaluation
        yield 1  # another estimate
        yield -(param["x"] ** 2)  # true objective function

    search = pyhopper.Search(
        {
            "x": pyhopper.float(),
        }
    )
    search.run(dummy_of, max_steps=5, quiet=True)

By default PyHopper uses the **last item** of the iterator as objective score and ignores the values yielded before.
However, the values yielded before can be used for detecting and consequently discontinuing unpromising evaluations.

To determine if an evaluation should be discontinued or not PyHopper provides the :meth:`pyhopper.pruners.Pruner` interface, which can be passed to calls of :code:`Search.run`.
For example

.. code-block:: python

    import pyhopper
    import numpy as np

    def noisy_objective(param):
        return -(param["x"] ** 2) + 0.1 * np.random.default_rng().normal()

    def generator_of(param):
        evals = []
        for i in range(5):
            evals.append(noisy_objective(param))
            yield np.mean(evals)                 # Current estimate of the objective

    search = pyhopper.Search(
        {
            "x": pyhopper.float(),
        }
    )
    search.run(generator_of, max_steps=50, pruner=pyhopper.pruners.QuantilePruner(0.8))

.. code-block:: text

    > Search is scheduled for 50 steps
    > Current best 0.0596: 100%|█████████████████████████| 50/50 [00:00<00:00, 101.45steps/s]
    > ======================= Summary ======================
    > Mode              : Best f : Steps : Pruned    : Time
    > -----------       : ---    : ---   : ---       : ---
    > Initial solution  : -4.09  : 1     : 0         : 9 ms
    > Random seeding    : 0.0596 : 6     : 43        : 60 ms
    > -----------       : ---    : ---   : ---       : ---
    > Total             : 0.0596 : 7     : 43        : 69 ms
    > ======================================================

discontinues evaluation if at least one of the intermediate results are within the *worse* 0.8-quantile of the non-pruned intermediate results so far.

For convenience, the :meth:`pyhopper.wrap_n_times` wrapper function accepts an optional argument :code:`yield_after` that turns the wrapped function in to a generator function.

.. code-block:: python

    import pyhopper
    import numpy as np

    def noisy_objective(param):
        return -(param["x"] ** 2) + 0.1 * np.random.default_rng().normal()

    search = pyhopper.Search(
        {
            "x": pyhopper.float(),
        }
    )
    search.run(
        pyhopper.wrap_n_times(noisy_objective, n=5, yield_after=0),
        max_steps=50,
        pruner=pyhopper.pruners.QuantilePruner(0.8),
    )

.. code-block:: text

    > Search is scheduled for 50 steps
    > Current best 0.0404: 100%|████████████████████████████| 50/50 [00:00<00:00, 99.08steps/s]
    > ======================== Summary =======================
    > Mode              : Best f   : Steps : Pruned    : Time
    > -----------       : ---      : ---   : ---       : ---
    > Initial solution  : -0.00734 : 1     : 0         : 10 ms
    > Random seeding    : 0.0404   : 5     : 44        : 49 ms
    > -----------       : ---      : ---   : ---       : ---
    > Total             : 0.0404   : 6     : 44        : 59 ms
    > ========================================================

A complete list of available pruners can be found at :ref:`pruning-api-label`.

Manually pruning evaluations
===============================

To manually prune a running evaluation we can raise a :meth:`pyhopper.PruneEvaluation` exception.

.. code-block:: python

    import pyhopper
    import numpy as np

    def noisy_objective(param):
        return -(param["x"] ** 2) + 0.1 * np.random.default_rng().normal()

    def generator_of(param):
        evals = []
        for i in range(5):
            value = noisy_objective(param)
            if value < -0.5:
                # Let's prune this evaluation if an evaluation is below -0.5
                raise pyhopper.PruneEvaluation()

            evals.append(value)
        return np.mean(evals)     # Final objective function

    search = pyhopper.Search(
        {
            "x": pyhopper.float(),
        }
    )
    search.run(generator_of, max_steps=50)

.. code-block:: text

    > Search is scheduled for 50 steps
    > Best f: 0.0388 (out of 32 params): 100%|███████|  [00:00<00:00, 2203.5 params/s]
    > ====================== Summary ======================
    > Mode              : Best f : Steps : Pruned   : Time
    > ----------------  : ----   : ----  : ----     : ----
    > Initial solution  : x      : 0     : 1        : 0 ms
    > Random seeding    : 0.0324 : 5     : 9        : 1 ms
    > Local sampling    : 0.0388 : 27    : 8        : 7 ms
    > ----------------  : ----   : ----  : ----     : ----
    > Total             : 0.0388 : 32    : 18       : 23 ms
    > =====================================================