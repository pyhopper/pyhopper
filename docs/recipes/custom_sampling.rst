Custom sampling strategies
-----------------------------

The default sampling strategies for PyHopper's built-in :meth:`pyhopper.float`, :meth:`pyhopper.int` and :meth:`pyhopper.choice` can be overwritten by custom procedures.
In particular, all of these three functions have the two arguments :code:`seeding_strategy` and :code:`mutation_strategy` to customize the seeding and local sampling strategies.

The :code:`seeding_strategy` takes no argument and expected to return a randomly sampled parameter candidate.
The :code:`mutation_strategy` is called with the current best parameter value as argument.

.. code-block:: python

    def dummy_of(param):
        return 0

    def seeding_fn():
        x = np.random.default_rng().normal()
        print(f"Seeding x = {x}")
        return x

    def mutation_fn(current_best):
        x = current_best + np.random.default_rng().normal()
        print(f"Mutating {current_best} to {x}")
        return x

    search = pyhopper.Search(
        {
            "gauss": pyhopper.float(
                -1, 1, seeding_strategy=seeding_fn, mutation_strategy=mutation_fn
            ),
        }
    )
    search.run(dummy_of, steps=3, quiet=True)

.. code-block:: text

    > Seeding x = -0.5400388703180178
    > Mutating -0.5400388703180178 to 0.3009240920850529
    > Mutating -0.5400388703180178 to -1.3059577801571272

Optionally, PyHopper will automatically detect if the passed mutation function accepts a second argument, in which case PyHopper passes the current temperature as second argument to the mutation function.
The temperature argument is a float that PyHopper decreases from 1 to 0 over the runtime of the local sampling stage.
Note that the temperature might not be strictly decreasing as PyHopper has some heuristics to slow down or even increase the temperature in case many duplicate candidates are sampled.

.. code-block:: python

    def mutation_with_temp(current_best, temp):
        print("Current temperature is ", temp)
        return current_best + temp * np.random.default_rng().normal()

    search = pyhopper.Search(
        {
            "gauss": pyhopper.float(-1, 1, mutation_strategy=mutation_with_temp),
        }
    )
    search.run(dummy_of, steps=5, quiet=True)

.. code-block:: text

    > Current temperature is  0.8
    > Current temperature is  0.8
    > Current temperature is  0.6
    > Current temperature is  0.4