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
from pyhopper.utils import parse_runtime


def test_parse_runtime():
    assert parse_runtime("1723") == 1723
    assert parse_runtime("1day") == 86400
    assert parse_runtime("1week") == 7 * 86400
    assert parse_runtime("2 week") == 2 * 7 * 86400
    assert parse_runtime("1d") == 86400
    assert parse_runtime("5d ") == 5 * 86400
    assert parse_runtime("5 days") == 5 * 86400
    assert parse_runtime("5 days 24h") == 6 * 86400
    assert parse_runtime("1d 2h") == 93600
    assert parse_runtime("2h 60min") == 10800
    assert parse_runtime("1h 30min") == 5400
    assert parse_runtime("1:30") == 90
    assert parse_runtime("1m 30s") == 90
    assert parse_runtime("1m 30sec") == 90
    assert parse_runtime("1min 30sec") == 90
    assert parse_runtime("1:1:30") == 3690
    assert parse_runtime("1:01:30") == 3690
    assert parse_runtime("1h 1m 30s") == 3690
    assert parse_runtime("1 h 1 min 30s") == 3690
    assert parse_runtime("1 h 1 m 30 s") == 3690


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


def of_kwargs(param):
    assert "a" in param.keys()
    assert "b" in param.keys()
    assert "c" in param.keys()
    return param["a"] * 0.5


def test_kwargs():
    search = pyhopper.Search(
        a=pyhopper.int(0, 10),
        b=pyhopper.float(0, 10),
        c="hello",
    )

    r1 = search.run(of_kwargs, direction="max", steps=10)
    assert "a" in r1.keys()
    assert "b" in r1.keys()
    assert "c" in r1.keys()


def test_merge():
    search = pyhopper.Search(
        {"a": pyhopper.int(0, 10)},
        {"b": pyhopper.int(0, 10)},
        c="hello",
    )

    r1 = search.run(of_kwargs, direction="max", steps=10)
    assert "a" in r1.keys()
    assert "b" in r1.keys()
    assert "c" in r1.keys()


def test_merge_raise():
    with pytest.raises(ValueError):
        pyhopper.merge_dicts({"a": pyhopper.int(0, 10)}, {"a": pyhopper.float(0, 10)})

    with pytest.raises(ValueError):
        search = pyhopper.Search(
            {"a": pyhopper.int(0, 10)},
            {"a": pyhopper.float(0, 10)},
            c="hello",
        )
    with pytest.raises(ValueError):
        search = pyhopper.Search(
            {"a": pyhopper.int(0, 10)},
            a="hello",
        )


def test_simple1():
    search = pyhopper.Search(
        {
            "lr": pyhopper.float(
                init=0.005,
                lb=1e-5,
                ub=1e-3,
            )
        }
    )

    r1 = search.run(of, direction="max", steps=10)
    r2 = search.run(of, direction="max", steps=50, kwargs={"x": 1})
    assert "lr" in r1.keys()
    assert "lr" in r2.keys()


def of_array(param, x=None):
    assert np.all(param["lr"] >= -10)
    assert np.all(param["lr"] <= 10)
    assert param["lr"].shape == (10, 10)
    return -np.sum(np.square(param["lr"] - 3))


def test_nparray():
    search = pyhopper.Search({"lr": pyhopper.float(lb=-10, ub=10, shape=(10, 10))})

    param = search.run(of_array, direction="max", steps=20)
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
        }
    )
    r1 = search.run(ofall, direction="max", steps=10)
    assert "lr" in r1.keys()
    assert "lr2" in r1.keys()
    assert "lr3" in r1.keys()
    assert "lr4" in r1.keys()
    assert "lr5" in r1.keys()
    r2 = search.run(ofall, direction="max", runtime=0.5)
    assert "lr" in r1.keys()
    assert "lr2" in r1.keys()
    assert "lr3" in r1.keys()
    assert "lr4" in r1.keys()
    assert "lr5" in r1.keys()


def test_float_register_name_negative_single_bound():
    with pytest.raises(ValueError):
        search = pyhopper.Search({"lr": pyhopper.float(-1)})


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
        }
    )
    r1 = search.run(ofall, direction="max", seeding_steps=5, steps=10)
    assert "lr" in r1.keys()
    assert "lr2" in r1.keys()
    assert "lr3" in r1.keys()


def test_int_register_name_negative_single_bound():
    with pytest.raises(ValueError):
        search = pyhopper.Search({"lr": pyhopper.int(-5)})
    with pytest.raises(ValueError):
        search = pyhopper.Search({"lr": pyhopper.int(0)})


def test_choice_register():
    search = pyhopper.Search(
        {
            "lr": pyhopper.choice([1, 2, 3]),
            "lr2": pyhopper.choice(["a", "b", "c"], "a"),
            "lr3": pyhopper.choice(["a", "b", "c"], is_ordinal=True),
        }
    )
    # r1 = search.run(of, seeding_runtime="1h", runtime="1s", n_jobs=2)
    r1 = search.run(of, direction="max", steps=10, n_jobs=5)
    r1 = search.run(of, direction="max", seeding_steps=2, steps=10, n_jobs=5)
    assert "lr" in r1.keys()


def slow_of(param):
    time.sleep(0.5)
    return -np.square(param["lr"] - 3e-4) * 10


def test_parallelization():
    search = pyhopper.Search(
        {
            "lr": pyhopper.choice([1, 2, 3]),
        }
    )
    start = time.time()
    r1 = search.run(of, direction="max", seeding_steps=4, steps=10, n_jobs=5)
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
        }
    )
    r1 = search.run(check_constraints_of, direction="min", steps=50, seeding_steps=30)


def of_freeze1(param):
    assert param["a"] == 5
    return param["a"] * param["b"]


def of_freeze2(param):
    assert param["a"] in [5, -20]
    return param["a"] * param["b"]


def of_freeze3(param):
    assert param["a"] in [5, 20], f"{param['a']}"
    return param["a"] * param["b"]


def of_freeze4(param):
    return param["a"] * param["b"]


def test_freeze():
    search = pyhopper.Search(
        {
            "a": pyhopper.int(-10, 10, init=5),
            "b": pyhopper.float(0, 1),
        }
    )
    search["a"] = search.best["a"]

    search.run(of_freeze1, direction="max", steps=15, seeding_steps=10)
    search["a"] = -20
    search.run(of_freeze2, direction="max", steps=15, seeding_steps=10)
    search["a"] = 20
    search.run(of_freeze3, direction="max", steps=15, seeding_steps=10)
    search.run(of_freeze4, direction="max", steps=15, seeding_steps=10)


def of_add(param):
    return param["a"] * param["b"]


def test_add_m():
    search = pyhopper.Search(
        {
            "a": pyhopper.int(-10, 10, init=5),
            "b": pyhopper.float(0, 1),
        }
    )
    search += {"b": 2}
    # search.sweep("a", [2, 5])
    search.run(of_add, direction="max", steps=15, seeding_steps=10)


def of_set(param):
    assert param["a"] == 3
    return param["a"] * param["b"]


def test_add_m2():
    search = pyhopper.Search(
        {
            "a": pyhopper.int(-10, 10, init=5),
            "b": pyhopper.float(0, 1),
        }
    )
    search["a"] = 3
    search.run(of_set, direction="max", steps=15, seeding_steps=10)


def of_options(param, x=None):
    assert "opt" in param.keys()
    assert param["opt"] in ["a", "b", "c"]
    return np.random.default_rng().normal()


def test_options():
    search = pyhopper.Search(opt=pyhopper.choice("a", "b", "c"))

    param = search.run(of_options, direction="max", steps=20)
    assert "opt" in param.keys()
    assert param["opt"] in ["a", "b", "c"]


if __name__ == "__main__":
    test_freeze()
