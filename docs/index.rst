.. PyHopper documentation master file, created by
   sphinx-quickstart on Tue Mar 16 13:09:22 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyHopper's documentation!
====================================


PyHopper is a hyperparam optimizer

.. code-block:: bash

    pip3 install -U pyhopper


.. code-block:: python

    x = Example()
    print(x.value)


User manual
--------------

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

.. py:class:: Foo

    .. py:method:: quux()

User's Guide
------------

This part of the documentation is about the user guide

.. toctree::
    :maxdepth: 1

    quickstart

API Reference
-------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
    :maxdepth: 2

    api