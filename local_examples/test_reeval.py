import numpy as np
import sys

sys.path.append("../PyHopper/")

import pyhopper.cancellers
import pyhopper
import matplotlib.pyplot as plt
import seaborn as sns


def of(config):
    return np.random.default_rng().random()


def test_simple1():
    search = pyhopper.Search(
        {
            "x": pyhopper.float(-10, 10),
            "y": pyhopper.float(-10, 10),
        },
        direction="max",
    )

    r1 = search.run(
        of, seeding_steps=10, max_steps=50, reevaluate_after_unsuccessful=20
    )

    sns.set()
    fig, ax = plt.subplots(figsize=(5, 5))

    b = ax.scatter(
        x=search.history.get_marginal("x"),
        y=search.history.get_marginal("y"),
        c=search.history.fs,
    )

    fig.colorbar(b, ax=ax)
    fig.tight_layout()
    fig.savefig("marginal.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.plot(search.history.steps, search.history.best_fs, color="red")
    ax.scatter(x=search.history.steps, y=search.history.fs, color="blue")
    fig.tight_layout()
    fig.savefig("search.png")
    plt.close(fig)


if __name__ == "__main__":
    test_simple1()