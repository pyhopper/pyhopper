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


from typing import Union
import numpy as np


class NTimesEvaluator:
    def __init__(self, func, n, yield_after=0, reduction="mean", pass_index_arg=False):
        if n <= 0:
            raise ValueError(f"n must be > 0, but got {n}")
        if yield_after is not None and yield_after >= n:
            raise ValueError(
                f"'yield_after' must be less than 'n', but got {yield_after} and {n}"
            )
        if callable(reduction):
            self._reduction = reduction
        elif reduction == "mean":
            self._reduction = np.mean
        elif reduction in ["med", "median"]:
            self._reduction = np.median
        else:
            raise ValueError(
                f"Unknown reduction '{reduction}'. Pass either 'mean','median', or a callable"
            )
        self._pass_index_arg = pass_index_arg
        self._n = n
        self._yield_after = yield_after
        self._func = func

    def __call__(self, param, **kwargs):
        results = []
        for i in range(self._n):
            if self._pass_index_arg:
                r = self._func(param, i, **kwargs)
            else:
                r = self._func(param, **kwargs)
            if r is None:
                raise ValueError(
                    f"Objective function returned None. The probably means you forgot to add a 'return' statement at "
                    f"the end of the function "
                )
            results.append(float(r))
            if self._yield_after is not None and i >= self._yield_after:
                yield self._reduction(results)
        if self._yield_after is None:
            return self._reduction(results)


def parse_timeout(timeout: Union[int, float, str]):
    if isinstance(timeout, float) or isinstance(timeout, int):
        return timeout
    if " " in timeout:
        # 5d 1h or 5d 1:0:0 pattern
        parts = timeout.split(" ")
        total_time = 0
        for part in parts:
            total_time += parse_timeout(part)
        return total_time
    elif ":" in timeout:
        # h:m:s or m:s pattern
        parts = timeout.split(":")
        total_time = 0
        for part in parts:
            total_time *= 60
            total_time += int(part)
        return total_time
    elif "d" in timeout:
        timeout = timeout.replace("days", "").replace("day", "").replace("d", "")
        if timeout.strip() == "":
            raise ValueError(
                "Could not parse number of days in timeout-string. Hint: no spaces are allowed between the number and "
                "the units, e.g., 3days "
            )
        return int(timeout) * 60 * 60 * 24
    elif "h" in timeout:
        timeout = timeout.replace("hours", "").replace("hour", "").replace("h", "")
        if timeout.strip() == "":
            raise ValueError(
                "Could not parse number of hours in timeout-string. Hint: no spaces are allowed between the number "
                "and the units, e.g., 12h "
            )
        return int(timeout) * 60 * 60
    elif "m" in timeout:
        # TODO: maybe just get rid of the non-digit characters
        timeout = (
            timeout.replace("minutes", "")
            .replace("minute", "")
            .replace("mins", "")
            .replace("min", "")
            .replace("m", "")
        )
        if timeout.strip() == "":
            raise ValueError(
                "Could not parse number of minutes in timeout-string. Hint: no spaces are allowed between the number "
                "and the units, e.g., 60min "
            )
        return int(timeout) * 60
    else:
        # TODO: maybe just get rid of the non-digit characters
        timeout = (
            timeout.replace("seconds", "")
            .replace("second", "")
            .replace("secs", "")
            .replace("sec", "")
            .replace("s", "")
        )
        if timeout.strip() == "":
            raise ValueError(
                "Could not parse number of seconds in timeout-string. Hint: no spaces are allowed between the number "
                "and the units, e.g., 10s "
            )
        return int(timeout)


def sanitize_bounds(lb, ub):
    if lb is not None and ub is None:
        if np.any(lb <= 0):
            raise ValueError(
                "Cannot register parameter. If only a single bound is provided it is treated as upper bound and the "
                "lower bound defaults to 0, but the provided bound is negative. Providing both bounds. "
            )
        ub = lb
        lb = 0
    if lb is None and ub is not None:
        if np.any(ub <= 0):
            raise ValueError(
                "Cannot register parameter. If only a single bound is provided it is treated as upper bound and the "
                "lower bound defaults to 0, but the provided bound is negative. Providing both bounds. "
            )
        lb = 0
    if lb is not None and ub is not None:
        temp = np.minimum(lb, ub)
        ub = np.maximum(lb, ub)
        lb = temp
    return lb, ub


def infer_shape(*args):
    # TODO: If there are multiple np.ndarray the shape should be the largest one while making sure the others are
    #  broadcastable
    shape = None
    for v in args:
        if isinstance(v, np.ndarray):
            shape = v.shape
    return shape


def steps_to_pretty_str(steps):
    if steps > 1e6:
        return f"{steps//1e6:0.0f}M"
    if steps > 1e3:
        return f"{steps//1e3:0.0f}k"
    return str(steps)


def time_to_pretty_str(elapsed):
    seconds = elapsed % 60
    elapsed = elapsed // 60  # now minutes
    minutes = int(elapsed % 60)
    elapsed = elapsed // 60  # now hours
    hours = int(elapsed % 24)
    days = int(elapsed // 24)

    if days == 1:  # 1 day 03:39:01 (h:m:s)
        return f"{days:d} day {hours:02d}:{minutes:02d}:{seconds:02.0f} (h:m:s)"
    elif days > 1:  # 3 days 03:39:01 (h:m:s)
        return f"{days:d} days {hours:02d}:{minutes:02d}:{seconds:02.0f} (h:m:s)"
    elif hours > 0:  # 03:39:01 (h:m:s)
        return f"{hours:02d}:{minutes:02d}:{seconds:02.0f} (h:m:s)"
    elif minutes > 0:  # 39:01 (m:s)
        return f"{minutes:02d}:{seconds:02.0f} (m:s)"
    elif seconds > 20:  # 27s
        return f"{seconds:02.0f} s"
    elif seconds < 1:  # 837ms
        return f"{1000*seconds:0.0f} ms"
    else:  # 9.83s
        return f"{seconds:0.02f} s"


if __name__ == "__main__":

    def print_t(inp):
        # print(f"{str(inp)} -> {parse_timeout(inp)}")
        print(f"assert parse_timeout('{str(inp)}') == {parse_timeout(inp)}")

    print_t(1723)
    print_t(934.0438)
    print_t("1d")
    print_t("1d 2h")
    print_t("2h 60min")
    print_t("1h 30min")
    print_t("1:30")
    print_t("1m 30s")
    print_t("1m 30sec")
    print_t("1min 30sec")
    print_t("1:1:30")
    print_t("1:01:30")
    print_t("1h 1m 30s")
    print(time_to_pretty_str(0.0004))
    print(time_to_pretty_str(0.004))
    print(time_to_pretty_str(0.04))
    print(time_to_pretty_str(0.4))
    print(time_to_pretty_str(4.4))
    print(time_to_pretty_str(47.4))
    print(time_to_pretty_str(474.4))
    print(time_to_pretty_str(4746.4))
    print(time_to_pretty_str(47467.4))
    print(time_to_pretty_str(474678.4))
    print(time_to_pretty_str(4746788.4))
    print(time_to_pretty_str(47467888.4))
    print(time_to_pretty_str(474678888.4))