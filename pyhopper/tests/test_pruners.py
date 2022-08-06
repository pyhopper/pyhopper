# Copyright 2021 Mathias Lechner and the PyHopper team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sys
import time
import pytest

import pyhopper
import pyhopper.pruners


def of_inc(param):
    return param["a"] + param["b"]


def test_simple_inc_dec():
    search = pyhopper.Search(
        {
            "a": pyhopper.float(lb=0, ub=10, init=2),
            "b": pyhopper.float(lb=0, ub=10, init=2),
        }
    )

    r1 = search.run(of_inc, direction="max", max_steps=10)
    assert of_inc(r1) >= 4
    search = pyhopper.Search(
        {
            "a": pyhopper.float(lb=0, ub=10, init=2),
            "b": pyhopper.float(lb=0, ub=10, init=2),
        }
    )

    r1 = search.run(of_inc, direction="min", max_steps=10)
    assert of_inc(r1) <= 4


def of(param, x=None):
    for i in range(10):
        yield np.random.default_rng().normal() - np.square(param["lr"] - 4) * 10


def test_simple1():
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(
                lb=0,
                ub=10,
            )
        }
    )

    r1 = search.run(of, direction="max", max_steps=10)
    r1 = search.run(
        of, direction="max", max_steps=100, pruner=pyhopper.pruners.TopKPruner(5)
    )
    r1 = search.run(
        of,
        direction="max",
        max_steps=200,
        n_jobs=5,
        pruner=pyhopper.pruners.TopKPruner(5),
    )


def test_simple2():
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(
                lb=0,
                ub=10,
            )
        }
    )

    with pytest.raises(ValueError):
        search = pyhopper.Search({"lr": pyhopper.float(-1)})
    r1 = search.run(of, direction="max", max_steps=10)
    r1 = search.run(
        of,
        direction="max",
        max_steps=100,
        pruner=pyhopper.pruners.QuantilePruner(50),
    )
    r1 = search.run(
        of,
        direction="max",
        max_steps=100,
        n_jobs=5,
        pruner=pyhopper.pruners.QuantilePruner(90),
    )
    r1 = search.run(
        of,
        direction="max",
        max_steps=100,
        pruner=pyhopper.pruners.QuantilePruner(0.5),
    )


of_counter = 0


def of_prune_first(param, x=None):
    global of_counter
    of_counter += 1
    if of_counter < 4:
        raise pyhopper.PruneEvaluation()
    return np.random.default_rng().normal()


def test_exception1():
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(
                lb=0,
                ub=10,
            )
        }
    )
    r1 = search.run(of_prune_first, direction="max", max_steps=10)
    assert "lr" in r1.keys()


def of_nan(param, x=None):
    global of_counter
    of_counter += 1
    if of_counter in [0, 1, 4, 10, 50]:
        return np.NaN
    return np.random.default_rng().normal()


def test_nan():
    global of_counter
    of_counter = 0
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(
                lb=0,
                ub=10,
            )
        }
    )
    with pytest.raises(ValueError):
        of_counter = 0
        r1 = search.run(of_nan, direction="max", max_steps=10)
    of_counter = 0
    r1 = search.run(of_nan, direction="max", ignore_nans=True, max_steps=10)
    with pytest.raises(ValueError):
        of_counter = 0
        r1 = search.run(
            of_nan,
            direction="max",
            ignore_nans=True,
            max_steps=100,
            pruner=pyhopper.pruners.QuantilePruner(0.5),
        )
    with pytest.raises(ValueError):
        of_counter = 0
        r1 = search.run(of_nan, direction="max", max_steps=200, n_jobs=5)
    of_counter = 0
    r1 = search.run(of_nan, direction="max", ignore_nans=True, n_jobs=5, max_steps=300)
    of_counter = 0
    assert "lr" in r1.keys()


def of_nan2(param, x=None):
    r = np.random.default_rng().normal()
    if r > 0.5:
        return np.NaN
    return r


def test_nan_simple():
    global of_counter
    of_counter = 0
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(
                lb=0,
                ub=10,
            )
        }
    )
    of_counter = 0
    r1 = search.run(of_nan, direction="max", ignore_nans=True, max_steps=10)
    of_counter = 0
    r1 = search.run(of_nan2, direction="max", ignore_nans=True, n_jobs=5, max_steps=200)


def test_topk():
    pruner = pyhopper.pruners.TopKPruner(3)
    pruner.direction = "max"

    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0, 0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0, 0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False

    pruner.append([12, 12, 12, 12, 10])
    pruner.append([20, 20, 20, 20, 20])
    pruner.append([0, 0, 0, 0, 0])
    pruner.append([12, 12, 10, 12, 10])
    pruner.append([12, 12, 12, 12, 10])
    pruner.append([12, 12, 12, 12, 10])
    assert pruner.should_prune([11]) == True
    assert pruner.should_prune([11, 11]) == True
    assert pruner.should_prune([11, 11, 11]) == True
    assert pruner.should_prune([10, 11, 11, 11]) == True
    assert pruner.should_prune([11, 11, 11, 11, 0]) == False
    assert pruner.should_prune([12]) == False
    assert pruner.should_prune([12, 12]) == False
    assert pruner.should_prune([12, 12, 12]) == False
    assert pruner.should_prune([12, 12, 12, 12]) == False
    assert pruner.should_prune([12, 12, 12, 12, 12]) == False
    pruner.append([13, 15, 20, 13, 30])
    pruner.append([15, 13, 20, 15, 30])
    pruner.append([20, 15, 20, 20, 30])
    pruner.append([13, 20, 13, 20, 30])
    assert pruner.should_prune([12]) == True
    assert pruner.should_prune([12, 12]) == True
    assert pruner.should_prune([12, 12, 12]) == True
    assert pruner.should_prune([12, 12, 12, 12]) == True
    assert pruner.should_prune([12, 12, 12, 12, 12]) == False
    assert pruner.should_prune([50]) == False
    assert pruner.should_prune([50, 20]) == False
    assert pruner.should_prune([50, 20, 20]) == False
    assert pruner.should_prune([50, 20, 20, 20]) == False
    assert pruner.should_prune([20, 20, 20, 20, 20]) == False


def test_quantile2():
    pruner = pyhopper.pruners.QuantilePruner(50)
    pruner.direction = "min"

    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0, 0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0, 0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False

    pruner.append([5, 5, 5, 5, 5])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([0, 0, 0, 0, 0])
    pruner.append([0, 0, 0, 0, 0])
    pruner.append([0, 0, 0, 0, 0])
    pruner.append([7, 7, 7, 7, 7])
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0, 0]) == False
    assert pruner.should_prune([10, 10, 10, 10, 10]) == False
    assert pruner.should_prune([10, 10, 10]) == True
    assert pruner.should_prune([10]) == True
    assert pruner.should_prune([10, 10]) == True


def test_quantile():
    pruner = pyhopper.pruners.QuantilePruner(50)
    pruner.direction = "max"

    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0, 0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0, 0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False
    assert pruner.should_prune([0, 0, 0]) == False

    pruner.append([5, 5, 5, 5, 5])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([0, 0, 0, 0, 0])
    pruner.append([7, 7, 7, 7, 7])
    pruner.append([0, 0, 0, 0, 0])
    pruner.append([12, 12, 10, 12, 10])
    pruner.append([12, 12, 12, 12, 10])
    pruner.append([12, 12, 12, 12, 10])
    pruner.append([12, 12, 12, 12, 10])
    assert pruner.should_prune([8]) == True
    assert pruner.should_prune([8, 10]) == True
    assert pruner.should_prune([8, 10, 0]) == True
    assert pruner.should_prune([8, 10, 0, 5]) == True
    assert pruner.should_prune([8, 10, 0, 5, 9]) == False
    assert pruner.should_prune([12]) == False
    assert pruner.should_prune([12, 12]) == False
    assert pruner.should_prune([8, 10, 10]) == False
    assert pruner.should_prune([8, 10, 10, 12]) == False
    assert pruner.should_prune([8, 10, 10, 10]) == True
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    pruner.append([20, 20, 50, 20, 20])
    assert pruner.should_prune([12]) == True
    assert pruner.should_prune([12, 12]) == True
    assert pruner.should_prune([8, 10, 30]) == True
    assert pruner.should_prune([8, 10, 30, 12]) == True
    assert pruner.should_prune([15, 15, 30, 15, 15]) == False
    pruner.append([50, 50])
    pruner.append([50])
    for i in range(20):
        pruner.append([50, 100, 50])
    assert pruner.should_prune([20]) == True
    assert pruner.should_prune([20, 20]) == True
    assert pruner.should_prune([100]) == False
    assert pruner.should_prune([100, 100]) == False


if __name__ == "__main__":
    test_nan_simple()