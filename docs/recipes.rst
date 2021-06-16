.. _recipes:

Recipes
==========

This is the quickstart guide of PyHopper

Walkthrough: 1 epoch MNIST
--------------

Callbacks
--------------

Callbacks can hook at the following six events

.. code-block:: python

    class MyCallback(pyhopper.callbacks.Callback):
        def on_search_start(self, search: pyhopper.Search):
            pass # Called at the beginning of the search process

        def on_evaluate_start(self, param: dict):
            pass # Called after parameter `param` is sampled

        def on_evaluate_end(self, param: dict, f: float):
            pass # Called after parameter `param` is evaluated to the value f

        def on_evaluate_cancelled(self, param: dict):
            pass # Called if the evaluation of parameter `param` is canceled (non promising)

        def on_new_best(self, new_best: dict, f: float):
            pass # Called if a new best parameter `new_best` with value f was found

        def on_search_end(self, history: pyhopper.History):
            pass # Called at the end of the search process