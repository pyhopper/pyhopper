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
from pyhopper.utils import parse_timeout


def test_parse_timeout():
    assert parse_timeout("1723") == 1723
    assert parse_timeout("1d") == 86400
    assert parse_timeout("1d 2h") == 93600
    assert parse_timeout("2h 60min") == 10800
    assert parse_timeout("1h 30min") == 5400
    assert parse_timeout("1:30") == 90
    assert parse_timeout("1m 30s") == 90
    assert parse_timeout("1m 30sec") == 90
    assert parse_timeout("1min 30sec") == 90
    assert parse_timeout("1:1:30") == 3690
    assert parse_timeout("1:01:30") == 3690
    assert parse_timeout("1h 1m 30s") == 3690


def of(param, x=None):
    return -np.square(param["lr"] - 3e-4) * 10


def ofall(param, x=None):
    assert "lr" in param.keys()
    assert "lr2" in param.keys()
    assert "lr3" in param.keys()
    if "np10" in param.keys():
        assert param["np10"].shape[0] == 10
    if "mul8" in param.keys():
        assert np.all(param["mul8"] % 8 == 0)
    if "mul8_t" in param.keys():
        assert np.all(param["mul8_t"] % 8 == 0)
    return -np.square(param["lr"] - 3e-4) * 10


def test_simple1():
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(
                init=0.005,
                lb=1e-5,
                ub=1e-3,
            )
        },
        direction="max",
    )

    r1 = search.run(of, max_steps=10)
    r2 = search.run(of, max_steps=50, kwargs={"x": 1})
    assert "lr" in r1.keys()
    assert "lr" in r2.keys()


def of_array(param, x=None):
    assert np.all(param["lr"] >= -10)
    assert np.all(param["lr"] <= 10)
    assert param["lr"].shape == (10, 10)
    return -np.sum(np.square(param["lr"] - 3))


def test_nparray():
    search = pyhopper.Search(
        {"lr": pyhopper.float(lb=-10, ub=10, shape=(10, 10))},
        direction="max",
    )

    param = search.run(of_array, max_steps=20)
    assert "lr" in param.keys()
    assert np.all(param["lr"] >= -10)
    assert np.all(param["lr"] <= 10)
    assert param["lr"].shape == (10, 10)


def test_float_register():
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(
                init=0.005,
                lb=1e-5,
                ub=1e-3,
            ),
            "lr2": pyhopper.float(),
            "lr3": pyhopper.float(0, 1),
            "test": "asjldkfj",
            "lr4": pyhopper.float(init=0.1),
            "np10": pyhopper.float(np.zeros(10), np.ones(10)),
            "lr5": pyhopper.float(0.0001, 1, log=True, init=0.1),
            "lr6": pyhopper.float(1),
        },
        direction="max",
    )
    r1 = search.run(ofall, max_steps=10)
    assert "lr" in r1.keys()
    assert "lr2" in r1.keys()
    assert "lr3" in r1.keys()
    assert "lr4" in r1.keys()
    assert "lr5" in r1.keys()
    search.freeze("lr3")
    r2 = search.run(ofall, timeout=0.5)
    assert "lr" in r1.keys()
    assert "lr2" in r1.keys()
    assert "lr3" in r1.keys()
    assert "lr4" in r1.keys()
    assert "lr5" in r1.keys()
    assert r1["lr3"] == r2["lr3"]


def test_float_register_name_negative_single_bound():
    with pytest.raises(ValueError):
        search = pyhopper.Search(
            {"lr": pyhopper.float(-1)},
            direction="max",
        )


def test_int_register():
    search = pyhopper.Search(
        {
            "lr": pyhopper.int(
                init=0,
                lb=-1,
                ub=3,
            ),
            "no_arg": 1,
            "lr2": pyhopper.int(0, 5),
            "lr3": pyhopper.int(5),
            "lr4": pyhopper.int(init=0),
            "np10": pyhopper.int(init=np.zeros(10)),
            "mul8": pyhopper.int(init=np.zeros(10), multiple_of=8),
            "mul8_t": pyhopper.int(8, 128, multiple_of=8),
        },
        direction="max",
    )
    r1 = search.run(ofall, seeding_steps=5, max_steps=10)
    assert "lr" in r1.keys()
    assert "lr2" in r1.keys()
    assert "lr3" in r1.keys()


def test_int_register_name_negative_single_bound():
    with pytest.raises(ValueError):
        search = pyhopper.Search(
            {"lr": pyhopper.int(-5)},
            direction="max",
        )
    with pytest.raises(ValueError):
        search = pyhopper.Search(
            {"lr": pyhopper.int(0)},
            direction="min",
        )


def test_choice_register():
    search = pyhopper.Search(
        {
            "lr": pyhopper.choice([1, 2, 3]),
            "lr2": pyhopper.choice(["a", "b", "c"], "a"),
            "lr3": pyhopper.choice(["a", "b", "c"], is_ordinal=True),
        },
        direction="max",
    )
    # r1 = search.run(of, seeding_timeout="1h", timeout="1s", n_jobs=2)
    r1 = search.run(of, seeding_timeout="1h", max_steps=10, n_jobs=5)
    assert "lr" in r1.keys()


def slow_of(param):
    time.sleep(0.5)
    return -np.square(param["lr"] - 3e-4) * 10


def test_parallelization():
    search = pyhopper.Search(
        {
            "lr": pyhopper.choice([1, 2, 3]),
        },
        direction="max",
    )
    start = time.time()
    r1 = search.run(of, seeding_timeout="1h", max_steps=10, n_jobs=5)
    assert time.time() - start < 3


def check_constraints_of(param):
    assert 0 <= param["i10"] <= 10
    assert -10 <= param["ipm10"] <= 10
    assert 0.0 <= param["f01"] <= 1.0
    assert -1 <= param["fpm"] <= 1.0
    assert 0.0 < param["log1"] <= 1.0
    assert 1e-6 <= param["e36"] <= 1e-3
    assert len(f"{param['g1']:0.0e}") <= 5
    return 0.0


def test_float_constraints():
    search = pyhopper.Search(
        {
            "i10": pyhopper.int(10),
            "ipm10": pyhopper.int(-10, 10),
            "f01": pyhopper.float(0, 1),
            "fpm": pyhopper.float(-0, 1),
            "log1": pyhopper.float(0.00001, 1, log=True),
            "e36": pyhopper.float(1e-6, 1e-3, log=True),
            "g1": pyhopper.float(1e-6, 1e-3, log=True, precision=1),
        },
        direction="min",
    )
    r1 = search.run(check_constraints_of, max_steps=50, seeding_steps=30)


if __name__ == "__main__":
    test_choice_register()