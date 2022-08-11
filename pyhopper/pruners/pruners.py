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


import numpy as np


_global_pruner_obj = None
_global_intermediate_results_list = []


def set_global_pruner(pruner):
    global _global_pruner_obj
    global _global_intermediate_results_list
    _global_pruner_obj = pruner
    _global_intermediate_results_list = []


def should_prune(of_estimate: float) -> bool:
    """Asks the pruner object passed to `search.run` if the current evaluation should be pruned (= stopped)

    :param of_estimate: An estimate of the true object function. For instance, this can be the training accuracy of a neural network
        after a few epochs.
    :return: True if the pruner thinks this parameter candidate is not worth continuing evaluating. False if no pruner
        object was passed to `search.run` or if the pruner determines the candidate is worth further evaluating.
    """
    global _global_pruner_obj
    global _global_intermediate_results_list
    if _global_pruner_obj is None:  # no Pruner object -> don't prune
        return False
    _global_intermediate_results_list.append(of_estimate)
    return _global_pruner_obj.should_prune(_global_intermediate_results_list)


def get_intermediate_results_list():
    global _global_intermediate_results_list
    return _global_intermediate_results_list


class Pruner:
    def __init__(self):
        self.direction = None

    # returns True if new is better than old
    def is_better_or_equal(self, new, old, reduction=None):
        old = np.array(old)
        new = np.array(new)
        if self.direction == "max":
            result = new >= old
        elif self.direction == "min":
            result = new <= old
        else:
            raise ValueError(
                "Internal error! Someone forgot to set .direction property! This should not happen."
            )
        if reduction is None:
            return result
        return reduction(result)

    def append(self, partial_results: list, was_pruned: bool):
        raise NotImplementedError()

    def should_prune(self, partial_results: list):
        raise NotImplementedError()

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass


class QuantilePruner(Pruner):
    def __init__(self, q, warmup=5):
        super().__init__()
        if 1 <= q < 100:
            q /= 100
        if not 0 < q < 1:
            raise ValueError(f"q must be between 0 and 1 (got {q})")
        self.q = q
        self.n = None
        self.warmup = warmup
        self.intermediates = None

    def append(self, partial_results: list, was_pruned: bool):
        if self.n is None:
            # First call -> initialize with empty lists
            self.n = len(partial_results)
            self.intermediates = [[] for i in range(self.n)]

        if len(partial_results) > self.n:
            raise ValueError(
                f"Runtime error! Partial results has length {len(partial_results)}, "
                f"while previous calls had at most {self.n}. "
                "Make sure the objective function yield the same amount of partial results on every call!"
            )

        for i in range(len(partial_results)):
            self.intermediates[i].append(partial_results[i])

    def should_prune(self, partial_results: list):
        if self.n is not None and len(partial_results) == self.n:
            # In case already all n intermediate values are obtained there is not need to prune anymore
            return False
        # First 'warmup' runs will not be pruned
        if self.intermediates is None:
            return False

        new_index = len(partial_results) - 1
        if len(self.intermediates[new_index]) < self.warmup:
            # make sure we have at least warmup samples before starting to prune
            return False
        q = self.q if self.direction == "max" else 1.0 - self.q
        quantile = np.quantile(self.intermediates[new_index], q)
        # prune if arrived partial result is worse than `q` quantile
        return not self.is_better_or_equal(partial_results[new_index], quantile)

    def state_dict(self):
        return {"n": self.n, "intermediates": self.intermediates}

    def load_state_dict(self, state_dict):
        self.n = state_dict["n"]
        self.intermediates = state_dict["intermediates"]


class TopKPruner(Pruner):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.n = None
        self.top_k_intermediates = []
        self.top_k_of = []

    def state_dict(self):
        return {
            "n": self.n,
            "top_k_intermediates": self.top_k_intermediates,
            "top_k_of": self.top_k_of,
        }

    def load_state_dict(self, state_dict):
        self.n = state_dict["n"]
        self.top_k_intermediates = state_dict["top_k_intermediates"]
        self.top_k_of = state_dict["top_k_of"]

    def append(self, partial_results: list, was_pruned: bool):
        if self.n is None:
            # First call -> initialize with empty lists
            self.n = len(partial_results)
            self.top_k_intermediates = [[] for i in range(self.n)]

        if len(partial_results) < self.n:
            # evaluation was pruned -> not interesting
            return
        elif len(partial_results) > self.n:
            raise ValueError(
                f"Runtime error! Partial results has length {len(partial_results)}, "
                f"while previous calls had at most {self.n}. "
                "Make sure the objective function yield the same amount of partial results on every call!"
            )

        of = partial_results[-1]
        if len(self.top_k_of) < self.k:
            # we had less than k evaluations so far -> append
            self.top_k_of.append(of)
            for i in range(self.n):
                self.top_k_intermediates[i].append(partial_results[i])
        else:
            # check if we should add it
            rank = self.is_better_or_equal(of, self.top_k_of)
            if not np.any(rank):
                return
            self.top_k_of.append(of)
            for i in range(self.n):
                self.top_k_intermediates[i].append(partial_results[i])
            if len(self.top_k_of) > self.k:
                # Overfull top_k lists -> remove the worst item
                worst_index = int(
                    np.argmin(self.top_k_of)
                    if self.direction == "max"
                    else np.argmax(self.top_k_of)
                )
                for i in range(self.n):
                    self.top_k_intermediates[i].pop(worst_index)
                self.top_k_of.pop(worst_index)

    def should_prune(self, partial_results: list):
        # First k runs will not be pruned
        if len(self.top_k_of) < self.k:
            return False
        if self.n is not None and len(partial_results) == self.n:
            # In case already all n intermediate values are obtained there is not need to prune anymore
            return False

        for i, r in enumerate(partial_results):
            reference_set = self.top_k_intermediates[i]
            is_better_than_any = self.is_better_or_equal(r, reference_set, np.any)
            if not is_better_than_any:
                return True
        return False