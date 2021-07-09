Custom parameter types
-----------------------------

The recommended way to implement custom parameter types is by subclassing :meth:`pyhopper.Parameter` and overwriting its :code:`sample` and :code:`mutate` methods.
An example of both options for :code:`torch.Tensor` type parameters is

.. code-block:: python

    import pyhopper
    import torch

    def dummy_of(param):
        print(param)
        return 0

    class TorchParameter(pyhopper.Parameter):
        def __init__(self, size):
            super().__init__()
            self.size = size

        def sample(self):
            return torch.normal(0, 1, size=self.size)

        def mutate(self, value, temperature: float):
            return value + temperature * torch.normal(0, 1, size=self.size)

    search = pyhopper.Search(
        {
            "torch_param": TorchParameter(size=(2, 2)),
        }
    )
    search.run(dummy_of, max_steps=5, quiet=True)

.. code-block:: text

    > {'torch_param': tensor([[ 0.7021,  0.5528], [ 0.3095, -0.1710]])}
    > {'torch_param': tensor([[1.1351, 0.2635], [0.8819, 0.8879]])}
    > {'torch_param': tensor([[ 0.5685,  1.1966], [ 0.2809, -0.0059]])}
    > {'torch_param': tensor([[ 0.7270,  0.9103], [ 0.1998, -0.3033]])}
    > {'torch_param': tensor([[0.7232, 0.1011], [0.0034, 0.0875]])}


Alternatively, we can use the :meth:`pyhopper.custom` parameter template and pass the :code:`seeding_fn` and :code:`mutation_fn` functional arguments:

.. code-block:: python

    def seeding_fn():
        return torch.normal(0, 1, size=(2, 2))

    def mutation_fn(current_best):
        return current_best + torch.normal(0, 1, size=(2, 2))

    search = pyhopper.Search(
        {
            "other_option": pyhopper.custom(seeding_fn, mutation_fn),
        }
    )
    search.run(dummy_of, max_steps=5, quiet=True)

.. code-block:: text


    > {'other_option': tensor([[ 1.2583, -0.3296], [ 0.5179,  1.3335]])}
    > {'other_option': tensor([[ 2.1317, -1.3902], [ 1.8776,  2.1398]])}
    > {'other_option': tensor([[-1.0080,  0.9658], [ 0.6401,  1.0016]])}
    > {'other_option': tensor([[-0.1377,  0.5059], [-1.1958,  1.3582]])}
    > {'other_option': tensor([[1.3503, 0.7410], [1.9922, 1.4300]])}

.. tip::

    To use PyHopper's built-in parameter type features for :code:`torch.Tensor` types, we could also just use the built-in NumPy parameters and `convert them to torch.Tensor objects <https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#numpy-array-to-tensor>`_.