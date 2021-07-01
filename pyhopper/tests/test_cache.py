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

from pyhopper.cache import EvaluationCache


def of(param, x=None):
    return -np.square(param["lr"] - 3e-4) * 10


def test_cache():
    cache = EvaluationCache()
    p1 = {"hallo": 1, "abc": 0.5}
    p2 = {"hallo": 1, "abc": 0.5}
    assert p1 not in cache
    assert p2 not in cache
    cache.stage(p1)
    assert p1 in cache
    assert p2 in cache
    cache.commit(p2, 0.5)
    assert p1 in cache
    assert p2 in cache

    f1 = {"hallo": 1, "abc": np.random.default_rng(123).random(size=(40, 40))}
    f2 = {"hallo": 1, "abc": np.random.default_rng(123).random(size=(40, 40))}
    assert f1 not in cache
    assert f2 not in cache
    cache.stage(f1)
    assert f1 in cache
    assert f2 in cache
    cache.commit(f2, 0.5)
    assert f1 in cache
    assert f2 in cache

    e1 = {
        "xyz": np.array([0.5, 0.3]),
        "abc": np.random.default_rng(234).integers(-20, 20, size=(40, 40)),
    }
    e2 = {
        "xyz": np.array([0.5, 0.3]),
        "abc": np.random.default_rng(234).integers(-20, 20, size=(40, 40)),
    }
    assert e1 not in cache
    assert e2 not in cache
    cache.stage(e1)
    assert e1 in cache
    assert e2 in cache
    cache.commit(e2, 0.5)
    assert e1 in cache
    assert e2 in cache

    b1 = {
        "dataset": "cifar100",
        "batch_size": 128,
        "epochs": 250,
        "entanglement_size": 3,
        "cutout_range": 8,
        "entanglement": 0.1,
        "dropout": 0.05,
        "base_lr": 0.2,
        "decay_lr": 0.5,
        "decay2_lr": 0.2,
        "l2_decay": 5e-05,
    }
    b2 = {
        "dataset": "cifar100",
        "batch_size": 128,
        "epochs": 250,
        "entanglement_size": 3,
        "cutout_range": 8,
        "entanglement": 0.1,
        "dropout": 0.05,
        "base_lr": 0.2,
        "decay_lr": 0.5,
        "decay2_lr": 0.2,
        "l2_decay": 5e-05,
    }
    assert b1 not in cache
    assert b2 not in cache
    cache.stage(b1)
    assert b1 in cache
    assert b2 in cache
    cache.commit(b2, 0.5)
    assert b1 in cache
    assert b2 in cache


def test_cache_tuple():
    cache = EvaluationCache()
    e1 = (1, 2, 3, 4)
    e2 = (70, 40, 30)
    e3 = (70, 40, 30)
    assert e1 not in cache
    assert e2 not in cache
    cache.stage(e1)
    assert e1 in cache
    assert e2 not in cache
    cache.stage(e2)
    assert e1 in cache
    assert e2 in cache
    assert e3 in cache
    cache.commit(e2, 0.5)
    assert e1 in cache
    assert e2 in cache
    assert e3 in cache
    cache.commit(e1, 0.5)
    assert e1 in cache
    assert e2 in cache
    assert e3 in cache


if __name__ == "__main__":
    test_cache_tuple()