Multiple validation splits
-----------------------------

PyHopper provides an easy way to perform cross-validation in the objective function with minimal code changes,
while at the same time use its :code:`pyhopper.cancellers` API to discontinue evaluations if candidates turn our unpromising after the first cross-validation step.

In particular, the :meth:`pyhopper.wrap_n_times` wrapper function has the optional argument :code:`pass_index_arg` that will pass an additional int argument to the the wrapped function indicating which
of the n steps the current function call corresponds to.
This additional argument can then be used to perform the training-validation split, which reduces overfitting as we do not rely on a single split while at the same time make sure each candidate is evaluated with the same splits.

.. code-block:: python

    import pyhopper
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_iris

    def noisy_objective(param, eval_index):
        print(f"x={param['x']}, eval_index={eval_index}")
        X, y = load_iris(return_X_y=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.33, random_state=eval_index  # use eval_index as random state
        )
        # ... train on X_train, y_train
        # ... validate on X_val, y_val
        return val_accuracy

    search = pyhopper.Search(
        {
            "x": pyhopper.float(),
        }
    )
    search.run(
        pyhopper.wrap_n_times(noisy_objective, n=3, yield_after=0, pass_index_arg=True),
        max_steps=3,
        canceller=pyhopper.cancellers.Quantile(0.8),
        quiet=True,
    )

.. code-block:: text

    > x=-0.2101340164769267, eval_index=0
    > x=-0.2101340164769267, eval_index=1
    > x=-0.2101340164769267, eval_index=2
    > x=-0.0846185419082141, eval_index=0
    > x=-0.0846185419082141, eval_index=1
    > x=-0.0846185419082141, eval_index=2