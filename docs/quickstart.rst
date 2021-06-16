.. _quickstart:

Quickstart
==========

This is the quickstart guide of PyHopper

PyHopper's search algorithm
--------------

PyHopper uses a 2-stage Markov chain Monte Carlo (MCMC) based optimization algorithm.
In the first **random seeding** stage, PyHopper randomly generates candidates from the entirety of the search space.
The purpose of this stage is to spot the most promising part of the search space quickly.

In the second **local sampling** stage, PyHopper takes the current best parameter and mutates it to obtain a slightly different candidate.
If the newly generated parameter is better than its predecessor, it is stored as the new best parameter.
Otherwise, it will be discarded. The randomness of the mutation step is modulated by a temperature parameter that decreases over the algorithm's runtime.

.. figure:: img/sampling.webp
   :align: center

    PyHopper's two MCMC sampling strategies applied to a 2D example problem

As evaluating a candidate is costly, a smart hashing routine takes care that no candidate is evaluated twice.
This 2-stage MCMC process allows PyHopper to explore and exploit parameter spaces with millions of dimensions efficiently.

Defining the search space
--------------

The central component of PyHopper is the `pyhopper.Search` class. Its constructor requires a `dict` object that
acts as as the hyperparameter *template* and defines the search space.
During runtime, PyHopper samples candidates by instantiating concrete hyperparameter from this template.
The resulting candidates are `dict` objects as well, with the only difference that PyHopper template types are replaced by a sample obtained from the MCMC core.

.. code-block:: python

    import pyhopper

    def of(param):
        print(param)
        return 0

    search = pyhopper.Search(
        {
            "my_const_param": "cifar10",
            "my_int_param": pyhopper.int(100,500),
            "my_float_param": pyhopper.float(0,0.4),
            "my_choice_param": pyhopper.choice(["adam","rmsprop","sgd"]),
        }
    )
    search.run(of,max_steps=5)

.. code-block::

    >>> {'my_const_param': 'cifar10', 'my_int_param': 442, 'my_float_param': 0.16690259272502092, 'my_choice_param': 'sgd'}
    >>> {'my_const_param': 'cifar10', 'my_int_param': 198, 'my_float_param': 0.21107963087889003, 'my_choice_param': 'adam'}
    >>> {'my_const_param': 'cifar10', 'my_int_param': 159, 'my_float_param': 0.09813299118201196, 'my_choice_param': 'adam'}
    >>> {'my_const_param': 'cifar10', 'my_int_param': 203, 'my_float_param': 0.19852373670299772, 'my_choice_param': 'adam'}

Hyperparameter templates
~~~~~~~~~~~~~~~

As shown above, PyHopper has three built-in template types: `int`, `float`, and `choice` :ref:`api:parameters`.





 .. toctree::
    :maxdepth: 2

.. py:function:: enumerate(sequence[, start=0])

   Return an iterator that yields tuples of an index and an item of the
   *sequence*. (And so on.)

.. py:class:: Bar

   Example test

   .. py:method:: Bar.quux()

      This is a simple method

   .. py:method:: Bar.__init__(a,c)

      this is the init function

      .. code-block:: python

          x = Bar(1,"hello")
          x.print()

.. py:class:: Foo4

   .. py:method:: quux