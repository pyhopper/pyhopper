.. _checkpointing-label:

Checkpointing
----------------------------------------------------------

We can store and load the internal state of a :meth:`pyhopper.Search` instance using the methods :meth:`pyhopper.Search.save` and :meth:`pyhopper.Search.load`.
The internal state includes

- The current best parameter and corresponding objective value
- The history of all evaluated (and pruned) candidates and corresponding objective values
- Optionally: The current progress if `save` is called in the middle of `search.run` (by a callback or via the `checkpoint_path` argument of `search.run`)

The internal state that can be saved and loaded **does not contain** the search space itself, i.e., the types and ranges of the parameter. 
This is because when continuing a saved search, we may want to modify the search space, for instance, by enlarging the range of a parameter.

.. code-block:: python

    search = pyhopper.Search( ... )

    search.run(...)
    
    # Saves the list of all evaluated candidates and current best one in a file
    search.save("run_completed.ckpt")

    # We can later load the file to continue the tuning process
    search.load("run_completed.ckpt")

    # We can also provide a directory
    file = search.save("my_directory/")
    # file = "my_directory/pyhopper_run_00000.ckpt"

    file2 = search.save("my_directory/")
    # file2 = "my_directory/pyhopper_run_00001.ckpt"

Running Pyhopper on pre-emptive (spot) instances
================================================================

The :meth:`pyhopper.Search.run` has an argument `checkpoint_path`, that, when provided, will continuously save the progress of the search.
If the file `checkpoint_path` already exists, Pyhopper will try to load it and resume the remaining search.

.. code-block:: python

    search = pyhopper.Search( ... )

    search.run(..., checkpoint_path="my_checkpoint.ckpt")
    
This functionality might be useful when running Pyhopper on a pre-emptive machine (spot instances) that can be shutdown at anytime.
Passing the `checkpoint_path` argument to :meth:`pyhopper.Search.run` ensures that we can resume the search without a loss of information.


Loading and storing pruner states
==================================================================

If a `checkpoint_path` argument and a pruner object is passed to :meth:`pyhopper.Search.run`, the internal state of the pruner will also be saved in the checkpoint.

When manually loading/storing checkpoints, the :meth:`pyhopper.Search.save` and :meth:`pyhopper.Search.load` both have an argument `pruner`, that, when provided, will aim to save and load the internal state of the pruner as well.

.. code-block:: python

    search = pyhopper.Search( ... )
    pruner = pyhopper.pruners.TopKPruner(10)

    search.run(..., pruner=pruner, checkpoint_path="my_checkpoint.ckpt")

    # .... 

    other_search = pyhopper.Search( ... )
    other_pruner = pyhopper.pruners.TopKPruner(10)
    other_search.load("my_checkpoint.ckpt", pruner=other_pruner)
    # Restores the search history and the internal state of the pruner
    