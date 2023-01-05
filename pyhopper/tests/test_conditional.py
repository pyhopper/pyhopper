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


def of(param):
    return -np.square(param["lr"] - 3e-4) * 10


def test_manual():
    search = pyhopper.Search(
            lr = pyhopper.float(1e-5, 1e-2, fmt="0.1g"),
            cond = pyhopper.cases(case1="abs",case2=pyhopper.int(0,10),case3=["s1",pyhopper.int(-10,0)])
        )

    r1 = search.run(of, direction="max", steps=30,seeding_steps=10)
    assert "lr" in r1.keys()

def test_manual2():
    search = pyhopper.Search(
            lr = pyhopper.float(1e-5, 1e-2, fmt="0.1g"),
            cond = pyhopper.cases({"case1": "abs","case2": pyhopper.int(0,10),"case3":["s1",pyhopper.int(-10,0)]})
        )

    r1 = search.run(of, direction="max", steps=30,seeding_steps=10)
    assert "lr" in r1.keys()

def test_fail():
    with pytest.raises(ValueError):
        search = pyhopper.Search(
                not_cond = pyhopper.cases("case1", pyhopper.int(0,10))
            )
    with pytest.raises(ValueError):
        search = pyhopper.Search(
                not_cond = pyhopper.choice("case1", pyhopper.int(0,10))
            )
    with pytest.raises(ValueError):
        search = pyhopper.Search(
                not_cond = pyhopper.choice("case1", ["case2",pyhopper.int(0,10)])
            )