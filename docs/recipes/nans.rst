Dealing with NaNs and exceptions
---------------------------------

**Not a number** (NaN) values occurring in the objective function can have two possible reasons:

- There is a bug in the objective function code
- The hyperparameter candidate is bad and causes an unstable training (e.g. exploding gradients)

To distinguish these two cases, PyHopper by default raises a :meth:`ValueError()` in case a NaN is returned by the objective function (= assuming the first cause).
In case NaN can occur naturally in the objective function you can pass the :code:`ignore_nans=True` flag to calls of :meth:`pyhopper.Search.run`.
This way PyHopper will ignore any NaN returned by the objective function and discard the parameter candidate.

.. code-block:: python

    import pyhopper

    def of(params):
        if params["x"]>0.9:
            return float("nan")
        return params["x"]

    search = pyhopper.Search({"x":pyhopper.float(0,1)})

    search.run(of,"max",max_steps=10)                  # raises a ValueError
    search.run(of,"max",max_steps=10,ignore_nans=True) # works fine

Numerical instabilities can also result in **exceptions** raise by some code inside the objective function.

.. code-block:: python

    def of(params):
        try:
            # some code that may raise an exception
        except (ValueError, ZeroDivisionError):
            # Tell Pyhopper to ignore this parameter candidate
            raise pyhopper.CancelEvaluation()
        return params["x"]

    search = pyhopper.Search({"x":pyhopper.float(0,1)})

    search.run(of,"max",max_steps=10)

.. note::

    Pyhopper does not catch any exception other than :meth:`pyhopper.CancelEvaluation()` to allow debugging the objective function.