# Copyright 2022 Mathias Lechner and the PyHopper team
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
import os

import numpy as np
import sys
import time
import pytest

import pyhopper
from pyhopper.utils import parse_timeout, store_dict, load_dict


def of(param):
    return -np.square(param["lr"] - 3e-4) * 10


def test_manual():
    search = pyhopper.Search(lr=None)

    search += {"lr": 0.1}
    search += {"lr": 0.2}
    search += {"lr": 0.3}

    r1 = search.run(of, direction="max", steps=search.manual_queue_count)
    assert "lr" in r1.keys()


def test_checkpoint():
    search = pyhopper.Search(lr=pyhopper.float(0, 1))

    search += {"lr": 0.1}
    search += {"lr": 0.2}
    search += {"lr": 0.3}

    checkpoint_path = "/tmp/ph.test"
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
    r1 = search.run(of, direction="max", steps=3, checkpoint_path=checkpoint_path)
    assert "lr" in r1.keys()
    ckpt = load_dict(checkpoint_path)
    assert ckpt["history"]["log_candidate"][0]["lr"] == 0.1
    assert ckpt["history"]["log_candidate"][1]["lr"] == 0.2
    assert ckpt["history"]["log_candidate"][2]["lr"] == 0.3
    search += {"lr": 0.4}
    search += {"lr": 0.5}
    search += {"lr": 0.6}

    r1 = search.run(of, direction="max", steps=3, checkpoint_path=checkpoint_path)
    assert len(search.history._log_candidate) == 6
    os.remove(checkpoint_path)


def dummy_of(config):
    yield config["lr"] - 2
    yield config["lr"] - 1
    yield config["lr"]


def test_checkpoint_pruner():
    search = pyhopper.Search(lr=pyhopper.float(0, 1))
    pruner = pyhopper.pruners.TopKPruner(3)
    search += {"lr": 0.1}
    search += {"lr": 0.4}
    search += {"lr": 0.5}
    search += {"lr": 0.2}
    search += {"lr": 0.3}

    checkpoint_path = "/tmp/ph.test"
    if os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)

    r1 = search.run(
        dummy_of,
        direction="max",
        steps=5,
        checkpoint_path=checkpoint_path,
        pruner=pruner,
    )
    assert "lr" in r1.keys()
    ckpt = load_dict(checkpoint_path)
    assert 0.4 in ckpt["pruner"]["top_k_of"]
    assert 0.3 in ckpt["pruner"]["top_k_of"]
    assert 0.5 in ckpt["pruner"]["top_k_of"]

    pruner2 = pyhopper.pruners.TopKPruner(3)
    search.load(checkpoint_path, pruner=pruner2)
    pruner2.top_k_intermediates[1][1] = pruner.top_k_intermediates[1][1]
    pruner2.top_k_intermediates[0][2] = pruner.top_k_intermediates[0][2]
    os.remove(checkpoint_path)


def test_load_store():
    test = {"a": 1, "b": np.array((0.1, 0.2))}
    print(test)

    filename = "/tmp/example.zip"
    store_dict(filename, test)
    obj = load_dict(filename)
    os.remove(filename)

    assert obj["a"] == test["a"]
    assert np.allclose(test["b"], obj["b"])


def of_ul(param):
    assert 1e-5 <= param["l"] <= 1e-1
    assert 1 <= param["u"] <= 10
    return 0


def test_fmt():
    search = pyhopper.Search(
        l=pyhopper.float(1e-5, 1e-2, ":0.5g"), u=pyhopper.float(1, 10, ":0.1f")
    )

    r1 = search.run(of_ul, direction="max", steps=20)


def of_nested(param):
    assert param["group1"]["x"] == 1
    return -np.square(param["group1"]["lr"] - 3e-4) * 10 + param["group2"]["d"]


def test_nested():
    search = pyhopper.Search(
        group1={"lr": pyhopper.float(0, 1), "x": 1},
        group2={"d": pyhopper.int(0, 10), "z": None},
        x=None,
    )
    r1 = search.run(of_nested, direction="max", steps=10)
    assert "lr" in r1["group1"].keys()


def test_nested_raise():
    search = pyhopper.Search(
        group1={"lr": pyhopper.float(0, 1), "x": 1}, group2={"d": pyhopper.int(0, 10)}
    )
    with pytest.raises(ValueError):
        search += {"group2": 1}
    search += {"group2": {"d": 1}}
    r1 = search.run(of_nested, direction="max", steps=10)
    assert "lr" in r1["group1"].keys()


if __name__ == "__main__":
    # test_checkpoint()
    # test_checkpoint_pruner()
    test_nested()