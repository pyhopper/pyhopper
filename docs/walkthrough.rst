=========================
PyHopper's Philosophy
=========================

What's so special about PyHopper?
--------------------------------------------------

There exist many powerful hyperparameter tuning frameworks out there. However, we created PyHopper because we weren't fully satisfied with any them.
In our experience they often showcase some fancy integration with other tools, at the cost of usability. PyHopper puts the user first.

- **You define when the results are ready, PyHopper takes care of the rest.** Different to many existing tuning tools that treat a timeout argument only as a termination signal, PyHopper carefully manages the exploration-exploitation tradeoff based on the provided timeout. You need your results by Monday? PyHopper will do exploration on Saturday and exploitation of Sunday.
- **A single algorithm is enough.** Many other tuning frameworks provide dozens of different tuning algorithms for the user to choose. While all these algorithm have their own use-cases where they excel, they also create a new meta-problem of deciding which algorithm is best suited for your task. So, instead of finding the optimal learning rate, we are now faced with the problem of finding the optimal tuning algorithm.
- **PyHopper is lightweight and hackable.** As an indirect consequence of the point above, other tuning frameworks often consist of thousands of lines of code with several layers of abstraction, making it difficult to implement custom types and sampling strategies. With PyHopper, you can implement custom parameter types and sampling strategies with just two lines of code.
- **Manual tuning naturally integrates with PyHopper.** For many ML problems, we often start with some manual hyperparameter tuning before running an automated tool. Moreover, form our experience we have some intuitions of what parameters could work. With PyHopper, you can guess some of the parameters. PyHopper then automatically fills the remaining parameters with the currently optimal values and evaluates your guess.
- **Why not Bayesian optimization?** Bayesian optimization methods are quite effective for tuning lower dimensional hyperparameters (say `less than 20 <https://arxiv.org/pdf/1807.02811.pdf>`_). However, for higher dimensional problems fitting an appropriate Bayesian estimate of the objective function becomes challenging due to an increasingly unfavorable ratio of feature dimensions vs training samples. For such high dimensional black-box optimization problems MCMC has been shown to provide supreme performance.

The PyHopper's simple user interface allows running hyperparameter searches with less than 5 minutes setup time and minimal adaptations of existing code.



PyHopper's search algorithm
----------------------------------------------------

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