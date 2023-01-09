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
import gzip
import os
import pickle
from enum import Enum
from typing import Union
import numpy as np


class ParamInfo:
    """Holds auxiliary information about a parameter candidate

    Attributes:
        ``type``     Enum specifying how the parameter was sampled (valid values are pyhopper.CandidateType.INIT, pyhopper.CandidateType.MANUALLY_ADDED, pyhopper.CandidateType.RANDOM_SEEDING, pyhopper.CandidateType.LOCAL_SAMPLING).

        ``sampled_at``  UNIX epoch timestamp when the parameter candidate was sampled.

        ``finished_at``  UNIX epoch timestamp when the evaluation of the candidate was finished.

        ``is_pruned``  Bool indicating if the candidate was pruned.
    """

    type = None
    sampled_at = None
    finished_at = None
    is_pruned = False
    is_nan = False

    def __init__(self, candidate_type, sampled_at):
        self.type = candidate_type
        self.sampled_at = sampled_at


class WrappedSample:
    def __init__(self, value, aux):
        self.value = value
        self.aux = aux


def unwrap_sample(sample):
    if isinstance(sample, WrappedSample):
        return unwrap_sample(sample.value)
    elif isinstance(sample, dict):
        return {k: unwrap_sample(v) for k, v in sample.items()}
    elif isinstance(sample, list):
        return [unwrap_sample(v) for v in sample]
    else:
        return sample


class Candidate:
    def __init__(self, value):
        self.value = value
        self.unwrapped_value = unwrap_sample(value)


def merge_dicts(*args):
    """Merges multiple dictionaries (``dict``s) into a single dictionary.
    Raises ValueError if a key is contained in two dicts with different values.

    :param args: Sequence of ``dict``s that
    :return: The merged ``dict``
    """
    new_dict = {}
    for d in args:
        for k, v in d.items():
            if k in new_dict.keys() and new_dict[k] != v:
                raise ValueError(
                    f"Could not merge dicts! The key '{k}' is contained in multiple dictionaries with different values"
                )
            new_dict[k] = v
    return new_dict


def convert_to_list(list_or_obj):
    if list_or_obj is None:
        return []
    if not isinstance(list_or_obj, list):
        # Convert single callback object to a list of size 1
        list_or_obj = [list_or_obj]
    return list_or_obj


def store_dict(filename, obj):
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f)


def load_dict(filename):
    with gzip.open(filename, "rb") as f:
        obj = pickle.load(f)
    return obj


def convert_to_checkpoint_path(checkpoint_path):
    actual_path = checkpoint_path
    if os.path.isdir(checkpoint_path):
        for i in range(100000):
            actual_path = os.path.join(checkpoint_path, f"pyhopper_run_{i:05d}.ckpt")
            if not os.path.isfile(actual_path):
                break
    return actual_path


class CandidateType(Enum):
    INIT = 0
    MANUALLY_ADDED = 1
    RANDOM_SEEDING = 2
    LOCAL_SAMPLING = 3


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


def _contains_number(text):
    for c in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
        if c in text:
            return True
    return False


def parse_runtime(runtime: Union[int, float, str]):
    if isinstance(runtime, float) or isinstance(runtime, int):
        return runtime
    orig_runtime = runtime
    if " " in runtime:
        # 5d 1h or 5d 1:0:0 pattern
        parts = runtime.split(" ")

        merged_parts = [parts[0]]
        for i in range(1, len(parts)):
            if not _contains_number(parts[i]):
                merged_parts[-1] += parts[i]
            else:
                merged_parts.append(parts[i])
        total_time = 0
        for part in merged_parts:
            total_time += parse_runtime(part)
        return total_time
    elif ":" in runtime:
        # h:m:s or m:s pattern
        parts = runtime.split(":")
        total_time = 0
        for part in parts:
            total_time *= 60
            total_time += int(part)
        return total_time
    elif "w" in runtime:
        runtime = runtime.replace("weeks", "").replace("week", "").replace("w", "")
        if runtime.strip() == "":
            raise ValueError(
                f"Could not parse substring '{orig_runtime}' while attempting to parse weeks in runtime-string. "
            )
        return int(runtime) * 60 * 60 * 24 * 7
    elif "d" in runtime:
        runtime = runtime.replace("days", "").replace("day", "").replace("d", "")
        if runtime.strip() == "":
            raise ValueError(
                f"Could not parse substring '{orig_runtime}' while attempting to parse days in runtime-string. "
            )
        return int(runtime) * 60 * 60 * 24
    elif "h" in runtime:
        runtime = runtime.replace("hours", "").replace("hour", "").replace("h", "")
        if runtime.strip() == "":
            raise ValueError(
                f"Could not parse substring '{orig_runtime}' while attempting to parse hours in runtime-string. "
            )
        return int(runtime) * 60 * 60
    elif "m" in runtime:
        runtime = (
            runtime.replace("minutes", "")
            .replace("minute", "")
            .replace("mins", "")
            .replace("min", "")
            .replace("m", "")
        )
        if runtime.strip() == "":
            raise ValueError(
                f"Could not parse substring '{orig_runtime}' while attempting to parse minutes in runtime-string. "
            )
        return int(runtime) * 60
    else:
        runtime = (
            runtime.replace("seconds", "")
            .replace("second", "")
            .replace("secs", "")
            .replace("sec", "")
            .replace("s", "")
        )
        if runtime.strip() == "":
            raise ValueError(
                f"Could not parse substring '{orig_runtime}' while attempting to parse second in runtime-string. "
            )
        return int(runtime)


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
    if steps is None:
        return "-"
    if steps > 1e6:
        return f"{steps//1e6:0.0f}M"
    if steps > 1e3:
        return f"{steps//1e3:0.0f}k"
    return str(steps)


def time_to_pretty_str(elapsed):
    if elapsed is None:
        return "-"
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
        # print(f"{str(inp)} -> {parse_runtime(inp)}")
        print(f"assert parse_runtime('{str(inp)}') == {parse_runtime(inp)}")

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