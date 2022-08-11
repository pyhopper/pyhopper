from traceback import format_exception

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import seaborn as sns

# sys.path.insert(0, ".")
import pyhopper


def of(param):
    raise ValueError("lkajslkfj")


if __name__ == "__main__":

    search = pyhopper.Search(
        {
            "lr": pyhopper.float(1e-5, 1e-2, fmt="0.1g"),
            "dr": pyhopper.float(0, 0.5, fmt="0.2f"),
        }
    )
    search.run(of, "max", steps=30, seeding_steps=30, n_jobs=2)

    # try:
    #
    #     of(None)
    # except:
    #     etype, value, tb = sys.exc_info()
    #     e = "".join(format_exception(etype, value, tb, 4096))
    #     print(e)
