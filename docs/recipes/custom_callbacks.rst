Custom Callbacks
-----------------------------

Callbacks can hook at the following six events

.. code-block:: python

    class MyCallback(pyhopper.callbacks.Callback):
        def on_search_start(self, search: pyhopper.Search):
            pass # Called at the beginning of the search process

        def on_evaluate_start(self, param: dict, info: pyhopper.ParamInfo):
            pass # Called after parameter `param` is sampled

        def on_evaluate_end(self, param: dict, f: float, info: pyhopper.ParamInfo):
            pass # Called after parameter `param` is evaluated to the value f

        def on_evaluate_cancelled(self, param: dict, info: pyhopper.ParamInfo):
            pass # Called if the evaluation of parameter `param` is canceled

        def on_new_best(self, new_best: dict, f: float, info: pyhopper.ParamInfo):
            pass # Called if a new best parameter `new_best` with value f was found

        def on_search_end(self, history: pyhopper.History):
            pass # Called at the end of the search process

The type :meth:`pyhopper.ParamInfo` stores information about the particular candidate such as *how it was sampled* (random seeding, local sampling, manually added), *when the candidate was sampled* (as a :code:`time` timestamp) and *when its evaluation was finished*.
For instance,

.. code-block:: python

    import datetime as dt
    import time

    class MyCallback(pyhopper.callbacks.Callback):
        def on_evaluate_start(self, param: dict, info: pyhopper.ParamInfo):
            print(
                f"{param['x']} sampled at {dt.datetime.fromtimestamp(info.sampled_at):%Y %B %d - %H:%m}"
            )

        def on_evaluate_end(self, param: dict, f: float, info: pyhopper.ParamInfo):
            print(
                f"{param['x']} finished at {dt.datetime.fromtimestamp(info.finished_at):%Y %B %d - %H:%m}"
            )
            print(f"Took {info.finished_at-info.sampled_at:0.1f} seconds")

    def of(param):
        time.sleep(1)
        return np.random.default_rng().random()


    search = pyhopper.Search(
        {
            "x": pyhopper.int(0, 10),
        },
    )
    r1 = search.run(
        of,
        "max",
        max_steps=3,
        quiet=True,
        callbacks=[MyCallback()],
    )

outputs

.. code-block:: text

    > 5 sampled at 2021 July 06 - 14:07
    > 5 finished at 2021 July 06 - 14:07
    > Took 1.0 seconds
    > 3 sampled at 2021 July 06 - 14:07
    > 3 finished at 2021 July 06 - 14:07
    > Took 1.0 seconds
    > 6 sampled at 2021 July 06 - 14:07
    > 6 finished at 2021 July 06 - 14:07
    > Took 1.0 seconds