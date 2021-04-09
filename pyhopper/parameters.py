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
import typing
from inspect import signature


def round_to_int(fvalue):
    if isinstance(fvalue, np.ndarray):
        return fvalue.astype(np.int32)
    return int(fvalue)


def call_with_temperature(func, value, temperature):
    sig = signature(func)
    if len(sig.parameters) == 1:
        return func(value)
    return func(value, temperature)


class Parameter:
    def __init__(self):
        pass

    def sample(self) -> typing.Any:
        raise NotImplementedError()

    def mutate(self, value: typing.Any, temperature: float) -> typing.Any:
        raise NotImplementedError()


class CustomParameter(Parameter):
    def __init__(self, init, mutation_strategy, sampling_strategy):
        super().__init__()
        self._init = init
        self._mutation_strategy = mutation_strategy
        self._sampling_strategy = sampling_strategy

    def sample(self):
        return self._sampling_strategy()

    def mutate(self, value, temperature: float):
        return call_with_temperature(self._mutation_strategy, value, temperature)


class FixedParameter(Parameter):
    def __init__(self, value):
        super().__init__()
        self._value = value

    def sample(self):
        return self._value

    def mutate(self, value, temperature: float):
        return self._value


class FrozenParameter(Parameter):
    def __init__(self, value, previous_parameter):
        super().__init__()
        self._value = value
        self._previous_parameter = previous_parameter

    def sample(self):
        return self._value

    def mutate(self, value, temperature: float):
        return self._value

    def get_old(self):
        return self._previous_parameter


class IntParameter(Parameter):
    def __init__(
        self, shape, lb, ub, init, multiple_of, mutation_strategy, sampling_strategy
    ):
        super().__init__()
        self._lb = lb
        self._ub = ub
        self._init = init
        self._multiple_of = multiple_of
        self._mutation_strategy = mutation_strategy
        self._sampling_strategy = sampling_strategy

        self._shape = shape

    def _cast_if_scalar(self, v):
        if self._shape is None:
            # Cast to python integer
            return int(v)
        return v

    def sample(self):
        if self._sampling_strategy is not None:
            # Custom sampling
            new_value = self._sampling_strategy()
        else:
            # Default bounded mode
            new_value = np.random.default_rng().integers(
                self._lb, self._ub, size=self._shape, endpoint=True
            )
        if self._multiple_of is not None:
            new_value -= new_value % self._multiple_of
        if self._lb is not None or self._ub is not None:
            new_value = np.clip(new_value, self._lb, self._ub)
        return self._cast_if_scalar(new_value)

    def mutate(self, value, temperature: float):
        if self._mutation_strategy is not None:
            # deep copy value in case mutation_strategy operates in-place
            if isinstance(value, np.ndarray):
                value = np.copy(value)
            new_value = call_with_temperature(
                self._mutation_strategy, value, temperature
            )
        elif self._lb is None or self._ub is None:
            # Unbounded mode
            spread = round_to_int(temperature * 2 ** 16) + 1
            new_value = value + np.random.default_rng().integers(
                -spread, spread, size=self._shape
            )
        else:
            # Default bounded mode
            spread = self._ub - self._lb
            new_value = value + round_to_int(
                temperature
                * 0.5
                * np.random.default_rng().integers(-spread, spread, size=self._shape)
            )
        if self._multiple_of is not None:
            new_value -= new_value % self._multiple_of

        if self._lb is not None or self._ub is not None:
            new_value = np.clip(new_value, self._lb, self._ub)
        return self._cast_if_scalar(new_value)


class ChoiceParameter(Parameter):
    def __init__(self, options, init, is_ordinal, mutation_strategy, sampling_strategy):
        super().__init__()
        self._options = options
        self._init = init
        self._is_ordinal = is_ordinal
        self._mutation_strategy = mutation_strategy
        self._sampling_strategy = sampling_strategy

    def sample(self):
        if self._sampling_strategy is not None:
            new_value = self._sampling_strategy()
        else:
            new_value = self._options[
                np.random.default_rng().integers(0, len(self._options))
            ]
        return new_value

    def mutate(self, value, temperature: float):
        if self._mutation_strategy is not None:
            new_value = call_with_temperature(
                self._mutation_strategy, value, temperature
            )
        elif self._is_ordinal:
            spread = int(len(self._options))
            index = self._options.index(value)
            new_index = index + int(
                temperature * 0.5 * np.random.default_rng().integers(-spread, spread)
            )
            new_index = np.clip(new_index, 0, len(self._options) - 1)
            new_value = self._options[new_index]
        else:
            new_value = self._options[
                np.random.default_rng().integers(0, len(self._options))
            ]
        return new_value


class FloatParameter(Parameter):
    def __init__(
        self,
        shape,
        lb,
        ub,
        init,
        log,
        precision,
        mutation_strategy,
        sampling_strategy,
    ):
        super().__init__()
        self._lb = lb
        self._ub = ub
        self._init = init
        self._log = log
        self._precision = precision
        self._mutation_strategy = mutation_strategy
        self._sampling_strategy = sampling_strategy

        self._shape = shape

    def _cast_if_scalar(self, v):
        if self._shape is None:
            # Cast to python float
            return float(v)
        return v

    def _round_and_clip(self, value):
        if self._precision is not None:
            if self._shape is not None:
                raise NotImplementedError(
                    "Setting the precision of an array parameter is not yet supported"
                )
            # Use string format function to round to significant decimal digits
            if self._log:
                # round the significant digits
                fmt_string = "{:0." + str(self._precision) + "g}"
            else:
                # round to decimal digits
                fmt_string = "{:0." + str(self._precision) + "f}"
            value = float(fmt_string.format(value))
        if self._lb is not None or self._ub is not None:
            value = np.clip(value, self._lb, self._ub)
        return value

    def sample(self):
        if self._sampling_strategy is not None:
            new_value = self._sampling_strategy()
        elif not self._log:
            if self._ub is None:
                # in unbounded mode we sample a Gaussian
                new_value = np.random.default_rng().normal(size=self._shape)
            else:
                new_value = np.random.default_rng().uniform(
                    self._lb, self._ub, size=self._shape
                )
        else:
            # Log-uniform
            new_value = np.exp(
                np.random.default_rng().uniform(
                    np.log(self._lb), np.log(self._ub), size=self._shape
                )
            )
            # now new_value is log-scaled in interval [lb,ub]
        new_value = self._round_and_clip(new_value)
        return self._cast_if_scalar(new_value)

    def mutate(self, value, temperature: float):
        if self._mutation_strategy is not None:
            # deep copy value in case mutation_strategy operates in-place
            if isinstance(value, np.ndarray):
                value = np.copy(value)
            new_value = call_with_temperature(
                self._mutation_strategy, value, temperature
            )
        elif not self._log:
            if self._ub is None:
                # in unbounded mode we will just add a Gaussian
                new_value = value + np.random.default_rng().normal(
                    scale=temperature + 1e-8, size=self._shape
                )
            else:
                spread = 0.5 * (self._ub - self._lb)
                new_value = value + temperature * np.random.default_rng().uniform(
                    -spread, spread, size=self._shape
                )
        else:
            # Temperature 1 means we multiply or divide our parameter by 10 (1000% change),
            # temperature 0 means we change by 1%
            factor = temperature * (10 - 1.01) + 1.01
            factor = np.random.default_rng().uniform(1, factor, size=self._shape)
            # Random bits to indicate if we should multiply or divide the parameter by our scale factor
            rand_mask = np.random.default_rng().integers(0, 2, size=self._shape)
            factor = factor * rand_mask + (1 - rand_mask) / factor
            # now new_value is log-scaled in interval [0,1]
            new_value = value * factor
            # now new_value is log-scaled in interval [lb,ub]

        new_value = self._round_and_clip(new_value)
        return self._cast_if_scalar(new_value)