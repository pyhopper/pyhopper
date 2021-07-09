Travelling Salesman Problem with PyHopper
-----------------------------

Pyhopper's flexibility allows us to implement heuristics for combinatorial optimization problems such as for the `Travelling Salesman Problem<https://en.wikipedia.org/wiki/Travelling_salesman_problem>`_ (TSP) with few lines of code.
For instance, the `2-opt <https://en.wikipedia.org/wiki/2-opt>`_ heuristic is a well known search algorithm for the TSP:

.. code-block:: python

    import numpy as np
    import pyhopper
    import matplotlib.pyplot as plt

    N = 50
    cities = np.random.default_rng(123456).uniform(-10, 10, size=(N, 2))

    def tsp_objective(solution):
        order = solution["order"]
        p = cities[order]
        cost = np.sum(np.linalg.norm(p[:-1] - p[1:], axis=1))
        cost += np.linalg.norm(p[0] - p[-1])  # make a roundtrip
        return cost

    def two_opt(order):
        order = np.copy(order)  # Don't overwrite current best order
        i, j = np.random.default_rng().integers(0, N, 2)
        # Swap city i with city j
        temp = order[i]
        order[i] = order[j]
        order[j] = temp
        return order

    search = pyhopper.Search(
        {
            "order": pyhopper.custom(
                seeding_fn=lambda: np.random.default_rng().permutation(N),
                mutation_fn=two_opt,
            )
        }
    )
    solution = search.run(
        tsp_objective,
        "min",
        timeout="30s",
    )
    # Let's plot the cities' location
    plt.scatter(cities[:, 0], cities[:, 1], marker="o", color="black")

    order = solution["order"]
    # Arrange cities in solution's ordering and make it a roundtrip
    x = [cities[order[i], 0] for i in range(N)] + [cities[order[0], 0]]
    y = [cities[order[i], 1] for i in range(N)] + [cities[order[0], 1]]
    plt.plot(x, y)
    plt.show()


.. image:: ../img/tsp.gif
   :align: center