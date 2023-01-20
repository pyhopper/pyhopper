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

from pyhopper.utils import WrappedSample


def cast_to_int(fvalue):
    if isinstance(fvalue, np.ndarray):
        return fvalue.astype(np.int64)
    return int(fvalue)


def call_with_temperature(func, value, temperature):
    sig = signature(func)
    if len(sig.parameters) == 1:
        return func(value)
    return func(value, temperature)


class Parameter:
    def __init__(self, init=None):
        self.initial_value = init

    def sample(self) -> typing.Any:
        raise NotImplementedError()

    def mutate(self, value: typing.Any, temperature: float) -> typing.Any:
        raise NotImplementedError()


# class ConditionalParameter(Parameter):
#     def __init__(self, kwargs):
#         super().__init__()
#         if len(kwargs) == 0:
#             raise ValueError("Must pass at least one case for a conditional parameter!")
#         self.cases = list(kwargs.keys())
#         self.values = kwargs
#         self.initial_value = self.cases[0], self.values[self.cases[0]]
#
#     def sample(self):
#         k = np.random.default_rng().choice(self.cases)
#         v = self.values[k]
#         return k, v
#
#     def mutate(self, value, temperature: float):
#         switch_case = np.random.default_rng().choice(
#             [True, False], p=[temperature, 1.0 - temperature]
#         )
#         if switch_case:
#             return self.sample()
#         return value


class CustomParameter(Parameter):
    def __init__(self, init, mutation_strategy, sampling_strategy):
        super().__init__(init)
        self._mutation_strategy = mutation_strategy
        self._sampling_strategy = sampling_strategy

    def sample(self):
        return self._sampling_strategy()

    def mutate(self, value, temperature: float):
        return call_with_temperature(self._mutation_strategy, value, temperature)


class IntParameter(Parameter):
    def __init__(
        self,
        shape,
        lb,
        ub,
        init,
        multiple_of,
        mutation_strategy,
        sampling_strategy,
    ):
        super().__init__()
        self._lb = lb
        self._ub = ub
        self._multiple_of = multiple_of
        self._mutation_strategy = mutation_strategy
        self._sampling_strategy = sampling_strategy

        self._shape = shape

        if init is None:
            init = self._round_to_multiple_of(int((ub + lb) / 2))
            init = self._cast_if_scalar(init)
        self.initial_value = init

    def _cast_if_scalar(self, v):
        if self._shape is None:
            # Cast to python integer
            return int(v)
        elif not isinstance(v, np.ndarray):
            return np.zeros(self._shape, dtype=np.int64) + v  # broadcast to shape
        return v

    def _round_to_multiple_of(self, v):
        if self._multiple_of is None:
            return v
        new_value = v + self._multiple_of // 2
        new_value -= new_value % self._multiple_of
        return new_value

    def sample(self):
        if self._sampling_strategy is not None:
            # Custom sampling
            new_value = self._sampling_strategy()
        else:
            # Integer is always bounded
            new_value = np.random.default_rng().integers(
                self._lb, self._ub, size=self._shape, endpoint=True
            )
        new_value = self._round_to_multiple_of(new_value)

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
        else:
            # Integer is always bounded
            spread = self._ub - self._lb
            new_value = value + cast_to_int(
                np.round(
                    temperature
                    * 0.5
                    * np.random.default_rng().integers(
                        -spread, spread, size=self._shape
                    )
                )
            )
        new_value = self._round_to_multiple_of(new_value)

        if self._lb is not None or self._ub is not None:
            new_value = np.clip(new_value, self._lb, self._ub)
        return self._cast_if_scalar(new_value)


class PowerOfIntParameter(IntParameter):
    def __init__(
        self,
        shape,
        lb,
        ub,
        init,
        power_of,
        multiple_of,
        mutation_strategy,
        sampling_strategy,
    ):
        super().__init__(
            shape, lb, ub, init, multiple_of, mutation_strategy, sampling_strategy
        )
        self._power_of = power_of
        self._log_param = IntParameter(
            shape, int(np.log2(lb)), int(np.log2(ub)), None, None, None, None
        )
        if init is None:
            init = self._log_param.initial_value
            init = 2**init
            init = self._round_to_multiple_of(init)
            init = self._cast_if_scalar(init)
            self.initial_value = init

    def sample(self):
        if self._sampling_strategy is not None:
            # Custom sampling
            new_value = self._sampling_strategy()
        else:
            new_value = self._log_param.sample()
            new_value = 2**new_value
        new_value = self._round_to_multiple_of(new_value)

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
        else:
            log_value = cast_to_int(np.log2(value))
            new_value = self._log_param.mutate(log_value, temperature)
            # breakpoint()
            new_value = 2**new_value
        new_value = self._round_to_multiple_of(new_value)

        if self._lb is not None or self._ub is not None:
            new_value = np.clip(new_value, self._lb, self._ub)
        return self._cast_if_scalar(new_value)


class ChoiceParameter(Parameter):
    def __init__(
        self, options, init_index, is_ordinal, mutation_strategy, sampling_strategy
    ):
        super().__init__()
        self._options = options
        self._is_ordinal = is_ordinal
        self._mutation_strategy = mutation_strategy
        self._sampling_strategy = sampling_strategy
        self._int_param = IntParameter(
            None, 0, len(options) - 1, None, None, None, None
        )
        if init_index is None:
            if is_ordinal:
                init_index = (len(options) - 1) // 2
            else:
                init_index = 0
        self.initial_value = WrappedSample(options[init_index], init_index)

    def sample(self):
        if self._sampling_strategy is not None:
            new_index = self._sampling_strategy()
        else:
            new_index = self._int_param.sample()

        return WrappedSample(value=self._options[new_index], aux=new_index)

    def mutate(self, value, temperature: float):
        if self._mutation_strategy is not None:
            new_index = call_with_temperature(
                self._mutation_strategy, value, temperature
            )
        elif self._is_ordinal:
            # Values are ordered/related -> prefer adjacent items
            new_index = self._int_param.mutate(value.aux, temperature)
        else:
            # Values are not ordered/related -> just pick any item
            new_index = self._int_param.sample()
        return WrappedSample(value=self._options[new_index], aux=new_index)


class FloatParameter(Parameter):
    def __init__(
        self,
        shape,
        lb,
        ub,
        init,
        precision,
        mutation_strategy,
        sampling_strategy,
    ):
        super().__init__()
        self._lb = lb
        self._ub = ub
        self._precision = precision
        self._mutation_strategy = mutation_strategy
        self._sampling_strategy = sampling_strategy

        self._shape = shape

        if init is None:
            if lb is None and ub is None:
                init = 0
            elif lb is None:
                init = ub
            elif ub is None:
                init = lb
            else:
                init = (lb + ub) / 2
            init = self._round_and_clip(init)
            init = self._cast_if_scalar(init)
        self.initial_value = init

    def _cast_if_scalar(self, v):
        if self._shape is None:
            # Cast to python float
            return float(v)
        elif not isinstance(v, np.ndarray):
            return np.zeros(self._shape, dtype=np.float32) + v  # broadcast to shape
        return v

    def _round(self, value):
        if self._precision is not None:
            if self._shape is not None:
                raise NotImplementedError(
                    "Setting the precision of an array parameter is not yet supported"
                )
            # Use string format function to round to significant decimal digits
            # round to decimal digits
            fmt_string = "{:0." + str(self._precision) + "f}"
            value = float(fmt_string.format(value))
        return value

    def _round_and_clip(self, value):
        value = self._round(value)
        if self._lb is not None or self._ub is not None:
            value = np.clip(value, self._lb, self._ub)
        return value

    def sample(self):
        if self._sampling_strategy is not None:
            new_value = self._sampling_strategy()
        else:
            if self._ub is None:
                # in unbounded mode we sample a Gaussian
                new_value = np.random.default_rng().normal(size=self._shape)
            else:
                new_value = np.random.default_rng().uniform(
                    self._lb, self._ub, size=self._shape
                )
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
        else:
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

        new_value = self._round_and_clip(new_value)
        return self._cast_if_scalar(new_value)


class LogSpaceFloatParameter(FloatParameter):
    def __init__(
        self,
        shape,
        lb,
        ub,
        init,
        precision,
        mutation_strategy,
        sampling_strategy,
    ):
        super().__init__(
            shape, lb, ub, init, precision, mutation_strategy, sampling_strategy
        )

        if lb <= 0.0:
            raise ValueError(
                f"Logarithmically scaled parameter must have a lower bound > 0 (got {str(lb)})"
            )
        self._log_param = FloatParameter(
            shape, np.log(lb), np.log(ub), None, None, None, None
        )
        if init is None:
            init = np.exp(self._log_param.initial_value)
            init = self._round_and_clip(init)
            init = self._cast_if_scalar(init)
        self.initial_value = init

    def _round(self, value):
        if self._precision is not None:
            if self._shape is not None:
                raise NotImplementedError(
                    "Setting the precision of an array parameter is not yet supported"
                )
            # Use string format function to round to significant decimal digits
            # round the significant digits
            fmt_string = "{:0." + str(self._precision) + "g}"
            value = float(fmt_string.format(value))
        return value

    def sample(self):
        if self._sampling_strategy is not None:
            # Custom sampling
            new_value = self._sampling_strategy()
        else:
            new_value = self._log_param.sample()
            new_value = np.exp(new_value)

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
        else:
            log_value = np.log(value)
            new_value = self._log_param.mutate(log_value, temperature)
            new_value = np.exp(new_value)

        new_value = self._round_and_clip(new_value)
        return self._cast_if_scalar(new_value)
