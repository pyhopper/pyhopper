NumPy parameters
-----------------------------

For NumPy array parameters, the functions :meth:`pyhopper.float` and :meth:`pyhopper.int` provide a :code:`shape` argument.
In default case :code:`shape = None` the parameter created are scalar types (Python :code:`int` and :code:`float` types).
If :code:`shape` is a tuple of integers the created parameter values will be of :code:`np.ndarray` type with :code:`dtype=np.int64` and :code:`dtype=np.float32` respectively.

.. code-block:: python

    import pyhopper

    def dummy_of(param):
        print(param)
        return 0

    search = pyhopper.Search(
        {
            "scalar": pyhopper.float(-1, 1),
            "1d": pyhopper.float(-1, 1, shape=3),
            "2d": pyhopper.float(-1, 1, shape=(2, 2)),
        }
    )
    search.run(dummy_of, steps=3, quiet=True)

produces

.. code-block:: text

    > {'scalar': -0.430359, '1d': array([0.53367, 0.80678, 0.10515]), '2d': array([[-0.75503,  0.28752],  [ 0.1958 ,  0.53757]])}
    > {'scalar': 0.443020, '1d': array([ 0.47137, -0.21797,  0.31202]), '2d': array([[-0.11824,  0.16386], [ 0.57913, -0.34669]])}
    > {'scalar': -0.158847, '1d': array([ 0.22458,  0.66483, -0.45764]), '2d': array([[ 0.40102, -0.29829], [-0.35151, -0.16981]])}

Same works for integers and in combination with constraints

.. code-block:: python

    search = pyhopper.Search(
        {
            "0d_int": pyhopper.int(0, 10),
            "1d_int": pyhopper.int(2, 16, shape=3, power_of=2),
            "2d_int": pyhopper.int(0, 20, shape=(2, 2), multiple_of=5),
        }
    )
    search.run(dummy_of, steps=3, quiet=True)

.. code-block:: text

    > {'0d_int': 8, '1d_int': array([ 8,  4, 16]), '2d_int': array([[15,  5], [20, 10]])}
    > {'0d_int': 9, '1d_int': array([ 8,  4, 16]), '2d_int': array([[15,  0], [15, 15]])}
    > {'0d_int': 6, '1d_int': array([16,  2,  8]), '2d_int': array([[20,  5], [15, 15]])}