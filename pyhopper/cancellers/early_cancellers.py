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


class EarlyCanceller:
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

    def append(self, partial_results: list):
        raise NotImplementedError()

    def should_cancel(self, partial_results: list):
        raise NotImplementedError()


class QuantileCanceller(EarlyCanceller):
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

    def append(self, partial_results: list):
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

    def should_cancel(self, partial_results: list):
        if self.n is not None and len(partial_results) == self.n:
            # In case already all n intermediate values are obtained there is not need to cancel anymore
            return False
        # First 'warmup' runs will not be cancelled
        if self.intermediates is None:
            return False

        new_index = len(partial_results) - 1
        if len(self.intermediates[new_index]) < self.warmup:
            # make sure we have at least warmup samples before starting to cancel
            return False
        q = self.q if self.direction == "max" else 1.0 - self.q
        quantile = np.quantile(self.intermediates[new_index], q)
        # Cancel if arrived partial result is worse than `q` quantile
        return not self.is_better_or_equal(partial_results[new_index], quantile)


class TopKCanceller(EarlyCanceller):
    def __init__(self, k):
        super().__init__()
        self.k = k
        self.n = None
        self.top_k_intermediates = []
        self.top_k_of = []

    def append(self, partial_results: list):
        if self.n is None:
            # First call -> initialize with empty lists
            self.n = len(partial_results)
            self.top_k_intermediates = [[] for i in range(self.n)]

        if len(partial_results) < self.n:
            # evaluation was cancelled -> not interesting
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

    def should_cancel(self, partial_results: list):
        # First k runs will not be cancelled
        if len(self.top_k_of) < self.k:
            return False
        if self.n is not None and len(partial_results) == self.n:
            # In case already all n intermediate values are obtained there is not need to cancel anymore
            return False

        for i, r in enumerate(partial_results):
            reference_set = self.top_k_intermediates[i]
            is_better_than_any = self.is_better_or_equal(r, reference_set, np.any)
            if not is_better_than_any:
                return True
        return False