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


def of(param):
    return -np.square(param["lr"] - 3e-4) * 10


def test_kwargs():
    search = (pyhopper.Search(),)

    search += {"lr": 0.1}
    search += {"lr": 0.2}
    search += {"lr": 0.3}

    r1 = search.run(of, direction="max", max_steps=3)
    assert "lr" in r1.keys()


if __name__ == "__main__":
    test_simple1()