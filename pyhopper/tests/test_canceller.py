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
import pyhopper.cancellers


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
        of, direction="max", max_steps=100, canceller=pyhopper.cancellers.TopK(5)
    )
    r1 = search.run(
        of,
        direction="max",
        max_steps=200,
        n_jobs=5,
        canceller=pyhopper.cancellers.TopK(5),
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
        of, direction="max", max_steps=100, canceller=pyhopper.cancellers.Quantile(50)
    )
    r1 = search.run(
        of,
        direction="max",
        max_steps=100,
        n_jobs=5,
        canceller=pyhopper.cancellers.Quantile(90),
    )
    r1 = search.run(
        of, direction="max", max_steps=100, canceller=pyhopper.cancellers.Quantile(0.5)
    )


of_counter = 0


def of_cancel_first(param, x=None):
    global of_counter
    of_counter += 1
    if of_counter < 4:
        raise pyhopper.CancelEvaluation()
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
    r1 = search.run(of_cancel_first, direction="max", max_steps=10)
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
            canceller=pyhopper.cancellers.Quantile(0.5),
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
    canceller = pyhopper.cancellers.TopK(3)
    canceller.direction = "max"

    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0, 0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0, 0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False

    canceller.append([12, 12, 12, 12, 10])
    canceller.append([20, 20, 20, 20, 20])
    canceller.append([0, 0, 0, 0, 0])
    canceller.append([12, 12, 10, 12, 10])
    canceller.append([12, 12, 12, 12, 10])
    canceller.append([12, 12, 12, 12, 10])
    assert canceller.should_cancel([11]) == True
    assert canceller.should_cancel([11, 11]) == True
    assert canceller.should_cancel([11, 11, 11]) == True
    assert canceller.should_cancel([10, 11, 11, 11]) == True
    assert canceller.should_cancel([11, 11, 11, 11, 0]) == False
    assert canceller.should_cancel([12]) == False
    assert canceller.should_cancel([12, 12]) == False
    assert canceller.should_cancel([12, 12, 12]) == False
    assert canceller.should_cancel([12, 12, 12, 12]) == False
    assert canceller.should_cancel([12, 12, 12, 12, 12]) == False
    canceller.append([13, 15, 20, 13, 30])
    canceller.append([15, 13, 20, 15, 30])
    canceller.append([20, 15, 20, 20, 30])
    canceller.append([13, 20, 13, 20, 30])
    assert canceller.should_cancel([12]) == True
    assert canceller.should_cancel([12, 12]) == True
    assert canceller.should_cancel([12, 12, 12]) == True
    assert canceller.should_cancel([12, 12, 12, 12]) == True
    assert canceller.should_cancel([12, 12, 12, 12, 12]) == False
    assert canceller.should_cancel([50]) == False
    assert canceller.should_cancel([50, 20]) == False
    assert canceller.should_cancel([50, 20, 20]) == False
    assert canceller.should_cancel([50, 20, 20, 20]) == False
    assert canceller.should_cancel([20, 20, 20, 20, 20]) == False


def test_quantile2():
    canceller = pyhopper.cancellers.Quantile(50)
    canceller.direction = "min"

    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0, 0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0, 0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False

    canceller.append([5, 5, 5, 5, 5])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([0, 0, 0, 0, 0])
    canceller.append([0, 0, 0, 0, 0])
    canceller.append([0, 0, 0, 0, 0])
    canceller.append([7, 7, 7, 7, 7])
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0, 0]) == False
    assert canceller.should_cancel([10, 10, 10, 10, 10]) == False
    assert canceller.should_cancel([10, 10, 10]) == True
    assert canceller.should_cancel([10]) == True
    assert canceller.should_cancel([10, 10]) == True


def test_quantile():
    canceller = pyhopper.cancellers.Quantile(50)
    canceller.direction = "max"

    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0, 0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0, 0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False
    assert canceller.should_cancel([0, 0, 0]) == False

    canceller.append([5, 5, 5, 5, 5])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([0, 0, 0, 0, 0])
    canceller.append([7, 7, 7, 7, 7])
    canceller.append([0, 0, 0, 0, 0])
    canceller.append([12, 12, 10, 12, 10])
    canceller.append([12, 12, 12, 12, 10])
    canceller.append([12, 12, 12, 12, 10])
    canceller.append([12, 12, 12, 12, 10])
    assert canceller.should_cancel([8]) == True
    assert canceller.should_cancel([8, 10]) == True
    assert canceller.should_cancel([8, 10, 0]) == True
    assert canceller.should_cancel([8, 10, 0, 5]) == True
    assert canceller.should_cancel([8, 10, 0, 5, 9]) == False
    assert canceller.should_cancel([12]) == False
    assert canceller.should_cancel([12, 12]) == False
    assert canceller.should_cancel([8, 10, 10]) == False
    assert canceller.should_cancel([8, 10, 10, 12]) == False
    assert canceller.should_cancel([8, 10, 10, 10]) == True
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    canceller.append([20, 20, 50, 20, 20])
    assert canceller.should_cancel([12]) == True
    assert canceller.should_cancel([12, 12]) == True
    assert canceller.should_cancel([8, 10, 30]) == True
    assert canceller.should_cancel([8, 10, 30, 12]) == True
    assert canceller.should_cancel([15, 15, 30, 15, 15]) == False
    canceller.append([50, 50])
    canceller.append([50])
    for i in range(20):
        canceller.append([50, 100, 50])
    assert canceller.should_cancel([20]) == True
    assert canceller.should_cancel([20, 20]) == True
    assert canceller.should_cancel([100]) == False
    assert canceller.should_cancel([100, 100]) == False


if __name__ == "__main__":
    test_nan_simple()