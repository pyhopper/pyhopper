import time

import numpy as np
import pyhopper
from pyhopper.callbacks.tensorboard import TensorboardCallback


def of(param):
    time.sleep(1)
    return np.random.default_rng().random()


def test_simple_tensorboard():
    search = pyhopper.Search(
        {
            "a": pyhopper.float(0, 1),
            "b": pyhopper.int(50, 100),
            "c": pyhopper.choice([0, 1, 2]),
        },
    )
    search.run(
        of,
        "max",
        "3min",
        callbacks=[
            TensorboardCallback("tb_example"),
        ],
    )


if __name__ == "__main__":
    test_simple_tensorboard()