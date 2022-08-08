import numpy as np
import sys
import time
import pytest

import pyhopper


def of_check(param):
    assert 2 ** (np.log2(param["pow2"])) == param["pow2"]
    assert param["mul5"] % 5 == 0

    assert param["arr_mul5"].dtype == np.int64, str(param["arr_mul5"].dtype)
    assert param["arr_pow2"].dtype == np.int64, str(param["arr_pow2"].dtype)
    for i in range(param["arr_pow2"].shape[0]):
        assert 2 ** (np.log2(param["arr_pow2"][i])) == param["arr_pow2"][i]

    for i in range(param["arr_mul5"].shape[0]):
        assert param["arr_mul5"][i] % 5 == 0

    assert np.all(param["lf_1d"] >= 0.05)
    assert np.all(param["lf_1d"] <= 50)
    assert np.all(param["lf_2d"] >= 0.05)
    assert np.all(param["lf_2d"] <= 50)
    assert param["lf_1d"].shape[0] == 10
    assert param["lf_2d"].shape[0] == 10
    assert param["lf_2d"].shape[1] == 10
    # print("arr_pow2", str(param["arr_pow2"]))
    print("pow2", str(param["pow2"]))
    return 0


def test_int_register():
    search = pyhopper.Search(
        {
            "pow2": pyhopper.int(8, 1024, power_of=2),
            "mul5": pyhopper.int(5, 50, multiple_of=5),
            "arr_pow2": pyhopper.int(8, 1024, shape=10, power_of=2),
            "arr_mul5": pyhopper.int(5, 50, shape=10, multiple_of=5),
            "lf_1d": pyhopper.float(0.05, 50, shape=10, log=True),
            "lf_2d": pyhopper.float(0.05, 50, shape=(10, 10), log=True),
        }
    )
    r1 = search.run(of_check, direction="max", steps=20)


if __name__ == "__main__":
    test_int_register()