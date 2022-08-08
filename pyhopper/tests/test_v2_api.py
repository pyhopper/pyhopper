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

    r1 = search.run(of, direction="max", steps=3)
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

    r1 = search.run(of, direction="max", steps=3, checkpoint_path=checkpoint_path)
    assert len(search.history._log_candidate) == 6
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


if __name__ == "__main__":
    test_checkpoint()