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

from .cache import EvaluationCache
from .parameters import (
    FloatParameter,
    IntParameter,
    ChoiceParameter,
    CustomParameter,
    Parameter,
    PowerOfIntParameter,
    LogSpaceFloatParameter,
)
from .parallel import execute, TaskManager, SignalListener
import numpy as np
from typing import Union, Optional, Any, Tuple
from types import FunctionType
from enum import Enum
import time

from tqdm.auto import tqdm

from .utils import (
    parse_timeout,
    sanitize_bounds,
    infer_shape,
    time_to_pretty_str,
    steps_to_pretty_str,
)


class CandidateType(Enum):
    INIT = 0
    MANUALLY_ADDED = 1
    RANDOM_SEEDING = 2
    LOCAL_SAMPLING = 3


class ScheduledRun:
    def __init__(
        self,
        max_steps=None,
        timeout=None,
        seeding_steps=None,
        seeding_timeout=None,
        seeding_ratio=None,
        start_temperature=1.0,
        end_temperature=0.0,
    ):
        if max_steps is not None and timeout is not None:
            raise ValueError(
                "Cannot specify both 'max_steps' and 'timeout' at the same time, one of the two must be None"
            )
        self._step_limit = max_steps
        self._timeout = None
        if timeout is not None:
            self._timeout = parse_timeout(timeout)
            # print(f"Parsed {timeout} to {self._timeout} seconds")
        self._start_time = time.time()
        self._step = 0
        self._temp_start_units = 0
        self._sigterm_received = 0
        self._start_temperature = start_temperature
        self._end_temperature = end_temperature

        if seeding_steps is not None and seeding_timeout is not None:
            raise ValueError(
                "Can only specify one of 'seeding_steps' and 'seeding_timeout' at the same time, one of the two must be None"
            )

        self._seeding_timeout = None
        self._seeding_max_steps = None

        if seeding_timeout is None and seeding_steps is None:
            # seeding_ratio is only valid if no other argument was set
            if self._step_limit is not None:
                # Max steps mode with seeding ratio provided
                self._seeding_max_steps = int(seeding_ratio * max_steps)
            elif self._timeout is not None:
                # Timeout mode with seeding ratio provided
                self._seeding_timeout = seeding_ratio * self._timeout
        else:
            if seeding_timeout is not None:
                self._seeding_timeout = parse_timeout(seeding_timeout)
            self._seeding_max_steps = seeding_steps
        self._seeding_ratio = seeding_ratio  # only needed for endless mode

        self._temp_start = None
        self._force_quit_callback = None
        self._original_sigint_handler = None
        self.reset_temperature()

    def signal_gradually_quit(self):
        self._sigterm_received += 1

    def increment_step(self):
        self._step += 1

    @property
    def unit(self):
        if self._timeout is not None:
            return "sec"
        else:
            return "steps"

    @property
    def step(self):
        return self._step

    @property
    def current_runtime(self):
        return time.time() - self._start_time

    @property
    def is_timeout_mode(self):
        return self._timeout is not None

    @property
    def is_mixed_endless(self):
        """
        True if in endless seeding+sampling mode
        """
        return (
            self._timeout is None
            and self._step_limit is None
            and self._seeding_timeout is None
            and self._seeding_max_steps is None
        )

    @property
    def endless_seeding_ratio(self):
        return self._seeding_ratio if self._seeding_ratio is not None else 0.2

    @property
    def is_endless(self):
        """
        True if in endless (sampling) mode
        """
        return self._timeout is None and self._step_limit is None

    @property
    def total_units(self):
        if self._step_limit is not None:
            # step-scheduled mode
            return self._step_limit
        elif self._timeout is not None:
            # time-scheduled mode
            return self._timeout
        return 1

    @property
    def current_units(self):
        if self._step_limit is not None:
            # step-scheduled mode
            return self._step
        elif self._timeout is not None:
            # time-scheduled mode
            return np.minimum(time.time() - self._start_time, self._timeout)
        return 0

    @property
    def current_runtime(self):
        return time.time() - self._start_time

    @property
    def is_disabled(self):
        if self._step_limit is not None:
            # step-scheduled mode
            return self._step_limit <= 0
        elif self._timeout is not None:
            # time-scheduled mode
            return self._timeout <= 0
        return False

    def is_timeout(self, estimated_runtime=0):
        if self._sigterm_received > 0:
            return True
        if self._step_limit is not None:
            # step-scheduled mode
            return self._step >= self._step_limit
        elif self._timeout is not None:
            # time-scheduled mode
            return time.time() - self._start_time + estimated_runtime >= self._timeout
        else:
            return False

    def is_seeding_timeout(self, estimated_runtime=0):
        if self._sigterm_received > 0:
            return True
        if self._seeding_max_steps is not None:
            # step-scheduled mode
            return self._step >= self._seeding_max_steps
        elif self._seeding_timeout is not None:
            # time-scheduled mode
            return (
                time.time() - self._start_time + estimated_runtime
                >= self._seeding_timeout
            )
        else:
            return False

    def reset_temperature(self):
        self._temp_start_units = self.current_units

    def to_elapsed_str(self):
        return (
            f"{self._step} steps ({time_to_pretty_str(time.time()-self._start_time)})"
        )

    def to_total_str(self):
        if self._step_limit is not None:
            return f"{self._step_limit} steps"
        elif self._timeout is not None:
            return f"{time_to_pretty_str(self._timeout)}"
        else:
            return "Endlessly (stop with CTRL+C)"

    @property
    def temperature(self):
        if self.is_endless:
            # In endless mode we randomly sample the progress
            progress = np.random.default_rng().random()
        else:
            progress = (self.current_units - self._temp_start_units) / max(
                self.total_units - self._temp_start_units, 1e-6  # don't divide by 0
            )
            progress = np.clip(progress, 0, 1)
        return (
            self._start_temperature
            + (self._end_temperature - self._start_temperature) * progress
        )


class History:
    """
    Public API for the history of the search. Can be used by the user for plotting and analyzing the search space.
    Persistent over several consecutive calls of ```run```
    """

    def __init__(self, keep_full_record=False):
        self._keep_full_record = keep_full_record
        self._log_candidate = []
        self._log_types = []
        self._log_f = []
        self._log_arrive_time = []
        self._log_best_f = []
        self._log_runtime = []

        self._cancelled_types = []
        self._cancelled_candidates = []
        self._cancelled_arrive_time = []
        self._cancelled_runtime = []
        self._start_time = time.time()

    def append_cancelled(self, candidate, candidate_type, runtime):
        self._cancelled_runtime.append(runtime)
        self._cancelled_types.append(candidate_type)
        self._cancelled_arrive_time.append(time.time() - self._start_time)
        if self._keep_full_record:
            self._cancelled_candidates.append(candidate)

    @property
    def keep_full_record(self):
        return self._keep_full_record

    def append(self, candidate, candidate_type, runtime, f, best_f):
        if self._keep_full_record:
            self._log_candidate.append(candidate)
        self._log_types.append(candidate_type)
        self._log_f.append(f)
        self._log_arrive_time.append(time.time() - self._start_time)
        self._log_best_f.append(best_f)
        self._log_runtime.append(runtime)

    def __getitem__(self, item):
        if not self._keep_full_record:
            raise ValueError(
                f"Error: Candidates were not recorded because ```keep_parameter_history``` argument passed to "
                f"```pyhopper.Search``` was set to False. "
            )
        return self._log_candidate[item]

    def get_marginal(self, item):
        if not self._keep_full_record:
            raise ValueError(
                f"Error: Candidates were not recorded because ```keep_parameter_history``` argument passed to "
                f"```pyhopper.Search``` was set to False. "
            )
        if len(self._log_candidate) > 0:
            if item not in self._log_candidate[0].keys():
                raise ValueError(
                    f"Error: Could not find key '{item}' in logged parameters"
                )
        return [self._log_candidate[i][item] for i in range(len(self._log_candidate))]

    def get_cancelled_marginal(self, item):
        if not self._keep_full_record:
            raise ValueError(
                f"Error: Candidates were not recorded because ```keep_parameter_history``` argument passed to "
                f"```pyhopper.Search``` was set to False. "
            )
        if len(self._cancelled_candidates) > 0:
            if item not in self._cancelled_candidates[0].keys():
                raise ValueError(
                    f"Error: Could not find key '{item}' in logged parameters"
                )
        return [
            self._cancelled_candidates[i][item]
            for i in range(len(self._cancelled_candidates))
        ]

    def __len__(self):
        return len(self._log_f)

    @property
    def fs(self):
        return self._log_f

    @property
    def best_f(self):
        return self._log_best_f[-1]

    @property
    def best_fs(self):
        return self._log_best_f

    @property
    def steps(self):
        return list(range(len(self._log_f)))

    @property
    def seconds(self):
        return self._log_arrive_time

    @property
    def minutes(self):
        return [t / 60 for t in self._log_arrive_time]

    @property
    def hours(self):
        return [t / 60 / 60 for t in self._log_arrive_time]

    def __repr__(self):
        repr_str = f"pyhopper.History(len={len(self)}"
        if len(self) > 0:
            repr_str += f", best={self.best_f:0.3g}"
        repr_str += ")"
        return repr_str

    def clear(self):
        self._log_candidate = []
        self._log_types = []
        self._log_f = []
        self._log_arrive_time = []
        self._log_best_f = []
        self._log_runtime = []

        self._cancelled_types = []
        self._cancelled_candidates = []
        self._cancelled_arrive_time = []
        self._cancelled_runtime = []
        self._start_time = time.time()


class LocalHistory:
    """
    Keeps track of internal statistics for each call of ```run```, i.e., what is printed at the end of run
    """

    def __init__(self, direction):
        self._direction = direction
        self.total_runtime = 0
        self.total_amount = 0
        self.total_cancelled = 0
        self.estimated_candidate_runtime = 0
        self.best_f = None

        self.best_per_type = {
            CandidateType.INIT: None,
            CandidateType.MANUALLY_ADDED: None,
            CandidateType.RANDOM_SEEDING: None,
            CandidateType.LOCAL_SAMPLING: None,
        }
        self.amount_per_type = {
            CandidateType.INIT: 0,
            CandidateType.MANUALLY_ADDED: 0,
            CandidateType.RANDOM_SEEDING: 0,
            CandidateType.LOCAL_SAMPLING: 0,
        }
        self.runtime_per_type = {
            CandidateType.INIT: 0,
            CandidateType.MANUALLY_ADDED: 0,
            CandidateType.RANDOM_SEEDING: 0,
            CandidateType.LOCAL_SAMPLING: 0,
        }
        self.cancelled_per_type = {
            CandidateType.INIT: 0,
            CandidateType.MANUALLY_ADDED: 0,
            CandidateType.RANDOM_SEEDING: 0,
            CandidateType.LOCAL_SAMPLING: 0,
        }

    def is_better(self, old, new):
        return (
            old is None
            or (self._direction == "max" and new > old)
            or (self._direction == "min" and new < old)
        )

    def append_cancelled(self, candidate_type):
        self.cancelled_per_type[candidate_type] += 1
        self.total_cancelled += 1

    def hot_start(self, best_f):
        self.best_f = best_f
        self.best_per_type[CandidateType.INIT] = best_f

    def append(self, candidate_type, runtime, f):
        if self.is_better(self.best_f, f):
            self.best_f = f
        self.total_amount += 1
        self.total_runtime += runtime
        if self.is_better(self.best_per_type[candidate_type], f):
            self.best_per_type[candidate_type] = f
        self.amount_per_type[candidate_type] += 1
        self.runtime_per_type[candidate_type] += runtime

        ema = 0.5
        self.estimated_candidate_runtime = (
            ema * self.estimated_candidate_runtime + (1 - ema) * runtime
        )


def register_int(
    lb: Optional[Union[int, float, np.ndarray]] = None,
    ub: Optional[Union[int, float, np.ndarray]] = None,
    init: Optional[Union[int, float, np.ndarray]] = None,
    multiple_of: Optional[int] = None,
    power_of: Optional[int] = None,
    shape: Optional[Union[int, Tuple]] = None,
    mutation_strategy: Optional[callable] = None,
    sampling_strategy: Optional[callable] = None,
) -> IntParameter:
    """Creates a new integer parameter

    :param lb: Lower bound of the parameter.
    :param ub: Upper bound of the parameter. If None, the `lb` argument will be used as upper bound with a lower bound of 0.
    :param init: Initial value of the parameter. If None it will be randomly sampled
    :param multiple_of: Setting this value to a positive integer enforces the sampled values of this parameter to be a mulitple of `multiple_of`.
    :param shape: For NumPy array type parameters, this argument must be set to a tuple containing the shape of the np.ndarray
    :param mutation_strategy: Setting this argument to a callable overwrites the default local sampling strategy. The callback gets called with the value
        of the the current best solution as argument and returns a mutated value
    :param sampling_strategy: Setting this argument to a callable overwrites the default random seeding strategy
    :return:
    """
    if lb is None and ub is None:
        # Unbounded int is actually a 32-bit integer
        lb = np.iinfo(np.int32).min
        ub = np.iinfo(np.int32).max
    lb, ub = sanitize_bounds(lb, ub)
    param_shape = infer_shape(shape, init, lb, ub) if shape is None else shape
    if power_of is not None:
        if power_of not in [1, 2]:
            raise ValueError(
                f"Power of {power_of} integers are currently not supported (only power 2 integers)."
            )
        return PowerOfIntParameter(
            shape,
            lb,
            ub,
            init,
            power_of,
            multiple_of,
            mutation_strategy,
            sampling_strategy,
        )
    param = IntParameter(
        param_shape,
        lb,
        ub,
        init,
        multiple_of,
        mutation_strategy,
        sampling_strategy,
    )
    return param


def register_custom(
    seeding_strategy: Optional[callable] = None,
    mutation_strategy: Optional[callable] = None,
    init: Any = None,
) -> CustomParameter:
    if seeding_strategy is None and init is None:
        raise ValueError(
            f"Could not create custom parameter, must either provide an initial value or a seeding strategy function"
        )
    param = CustomParameter(init, mutation_strategy, seeding_strategy)
    return param


def register_choice(
    options: list,
    init: Optional[Any] = None,
    is_ordinal: bool = False,
    mutation_strategy: Optional[FunctionType] = None,
    seeding_strategy: Optional[FunctionType] = None,
) -> ChoiceParameter:
    """Creates a new choice parameter

    :param options: List containing the possible values of this parameter
    :param init: Initial value of the parameter. If None it will be randomly sampled.
    :param is_ordinal: Flag indicating whether two neighboring list items ordered or not. If True, in the local sampling stage list items neighboring the current best value will be preferred. For sets with a natural ordering it is recommended to set this flag to True.
    :param mutation_strategy: Setting this argument to a callable overwrites the default local sampling strategy. The callback gets called with the value
        of the the current best solution as argument and returns a mutated value
    :param seeding_strategy: Setting this argument to a callable overwrites the default random seeding strategy
    :return:
    """
    if len(options) == 0:
        raise ValueError("List with possible values must not be empty.")
    param = ChoiceParameter(
        options, init, is_ordinal, mutation_strategy, seeding_strategy
    )
    return param


def register_float(
    lb: Optional[Union[int, float, np.ndarray]] = None,
    ub: Optional[Union[int, float, np.ndarray]] = None,
    init: Optional[Union[int, float, np.ndarray]] = None,
    log: Union[bool] = False,
    precision: Optional[int] = None,
    shape: Optional[Union[int, Tuple]] = None,
    mutation_strategy: Optional[FunctionType] = None,
    seeding_strategy: Optional[FunctionType] = None,
) -> FloatParameter:
    """Creates a new floating point parameter

    :param lb: Lower bound of the parameter. If both `lb` and `ub` are None, this parameter will be unbounded (usually not recommended).
    :param ub: Upper bound of the parameter. If None, the `lb` argument will be used as upper bound with a lower bound of 0.
    :param init: Initial value of the parameter. If None it will be randomly sampled
    :param shape: For NumPy array type parameters, this argument must be set to a tuple containing the shape of the np.ndarray
    :param log: Whether to use logarithmic or linearly scaling of the parameter.
        Defaults to False which searches the space linearly.
        If True, a logarithmic scaling is applied to the search space of this variable
    :param precision: Rounds the values to the specified significant digits.
        Defaults to None meaning that no rounding is applied
    :param mutation_strategy: Setting this argument to a callable overwrites the default local sampling strategy. The callback gets called with the value
        of the the current best solution as argument and returns a mutated value
    :param seeding_strategy: Setting this argument to a callable overwrites the default random seeding strategy
    """
    lb, ub = sanitize_bounds(lb, ub)
    if log and (lb is None or ub is None):
        raise ValueError(
            "Logarithmically distributed mode without bounds is not supported. Please specify lower and upper bound."
        )
    if log and (lb <= 0 or ub <= 0):
        raise ValueError(
            "Both bounds for logarithmically distributed parameter must be positive."
        )

    param_shape = infer_shape(init, lb, ub) if shape is None else shape
    if log:
        return LogSpaceFloatParameter(
            param_shape, lb, ub, init, precision, mutation_strategy, seeding_strategy
        )
    param = FloatParameter(
        param_shape,
        lb,
        ub,
        init,
        precision,
        mutation_strategy,
        seeding_strategy,
    )
    return param


class ProgBar:
    def __init__(self, schedule, run_history, disable):
        self._schedule = schedule
        self._run_history = run_history

        if self._schedule.is_endless:
            bar_format = (
                "Endless (stop with CTRL+C) {bar}| [{elapsed}<{remaining}{postfix}]",
            )
        elif self._schedule.is_timeout_mode:
            bar_format = "{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]"
        else:
            # step mode
            bar_format = "{l_bar}{bar}|  [{elapsed}<{remaining}{postfix}]"
        self._tqdm = tqdm(
            total=self._schedule.total_units,
            disable=disable,
            unit="",
            bar_format=bar_format,
        )

    def update(self, current_best):
        self._tqdm.n = self._schedule.current_units
        self._tqdm.set_postfix_str(self._str_time_per_eval(), refresh=False)
        if current_best is not None:
            self._tqdm.set_description_str(
                f"Best f: {current_best:0.3g} (out of {self._run_history.total_amount} params)",
                refresh=False,
            )
        self._tqdm.refresh()

    # Endless mode:
    # Endless (stop with CTRL+C)      best: 0.42 (out of 3213) [2.3min/param]
    # Step
    # 48% xXXXXXXXxxxxxxxxxxxxxxxxxxxxx | best: 0.42 (out of 3213) (00:38<1:00) [2.3min/param]
    # Time mode
    # 48% xXXXXXXXxxxxxxxxxxxxxxxxxxxxx | best: 0.42 (out of 3213) (00:38<1:00) [2.3min/param]
    def _str_time_per_eval(self):
        total_params_evaluated = (
            self._run_history.total_amount + self._run_history.total_cancelled
        )
        if total_params_evaluated == 0:
            return "..."
        seconds_per_param = self._schedule.current_runtime / total_params_evaluated
        if seconds_per_param <= 1:
            return f"{1/seconds_per_param:0.1f} params/s"
        elif seconds_per_param > 60 * 60:
            return f"{seconds_per_param/(60*60):0.1f} h/param"
        elif seconds_per_param > 60:
            return f"{seconds_per_param/60:0.1f} min/param"
        else:
            return f"{seconds_per_param:0.1f} s/param"

    def close(self, final_best):
        self.update(final_best)
        self._tqdm.close()


class Search:
    def __init__(
        self,
        parameters: dict,
        keep_parameter_history: bool = True,
    ):
        """
        Creates a new search object
        :param parameters: dict defining the search space
        :param keep_parameter_history: Whether to keep a copy of all evaluated parameters for later analysis.
            In case the parameter space contain large objects (e.g., Numpy arrays), it is recommended to set this
            value to False to reduce the memory footprint.
        """
        self._params = {}
        self._best_solution = {}
        self._free_params = {}
        for k, v in parameters.items():
            self._register_parameter(k, v)
        self._best_f = None
        self._canceller = None
        self._ignore_nans = None
        self._direction = None

        self._hooked_callbacks = []
        self._history = History(keep_parameter_history)
        self._run_history = None
        self._f_cache = EvaluationCache()
        self._manually_queued_candidates = []

        self._signal_listener = SignalListener()
        self._schedule = []
        self._objective_hash = None
        self._task_executor = None
        self._current_run_config = None

    def __iadd__(self, other):
        self.add(other)
        return self

    def __setitem__(self, key, value):
        if key in self._free_params.keys():
            del self._free_params[key]
        self._register_parameter(key, value)

    def _register_parameter(self, name: str, param: Any) -> None:
        if isinstance(param, Parameter):
            self._params[name] = param
            if self._best_solution.get(name) is None:
                self._best_solution[name] = param.initial_value
            self._free_params[name] = param
        else:
            self._params[name] = param
            self._best_solution[name] = param

    def overwrite_best(self, candidate: dict, f: Optional[float] = None) -> None:
        """Overwrites the current best solution with the provided parameter and objective function value

        :param candidate: Parameter values that will be set as current best candidate
        :param f: Objective function value that will be set as the current best value
        """
        for k, v in self._params.items():
            cv = candidate.get(k)
            if cv is not None:
                self._best_solution[k] = cv
            else:
                init = self._best_solution.get(k)
                if init is None:
                    raise ValueError(f"Parameter '{k}' has no initial value.")
        self._best_f = f

    def _update_free_params(self):
        free_params = []
        for k, v in self._params.items():
            if isinstance(v, Parameter):
                free_params.append(k)
        return free_params

    def forget_cached(self, candidate: dict):
        """Removes the given parameter candidate from the evaluation cache. This might be useful if a parameter value should be reevaluated.

        :param candidate: Parameter candidate to be wiped from the evaluation cache
        """
        self._f_cache.forget(candidate)

    def clear_cache(self):
        """Forgets all values of already evaluated parameters."""
        self._f_cache.clear()

    def add(self, candidate: dict) -> None:
        """
        Adding a guess for the optimal parameters to the search queue.
        :param candidate: dict representing a subset of the parameters assigned to a value
        """
        added_candidate = {}
        for k, v in self._params.items():
            cv = candidate.get(k)
            if cv is not None:
                added_candidate[k] = cv
            else:
                init = self._best_solution.get(k)
                if init is not None:
                    added_candidate[k] = init
                else:
                    raise ValueError(
                        f"Parameter '{k}' has no initial value and is not provided by the candidate solution."
                    )
        # Find typos and other bugs
        for k in candidate.keys():
            if k not in self._params:
                raise ValueError(
                    f"Candidate for '{k}' was provided by the candidate solution but has not been registered."
                )
        self._manually_queued_candidates.append(added_candidate)

    def sweep(self, name: str, candidate_values: list) -> None:
        """

        :param name:
        :param candidate_values:
        """
        if name not in self._params.keys():
            raise ValueError(f"Could not find '{name}' in set of registered parameters")
        for value in candidate_values:
            added_candidate = {}
            for k, v in self._params.items():
                if k == name:
                    added_candidate[k] = value
                    continue
                init = self._best_solution.get(k)
                if init is not None:
                    added_candidate[k] = init
                else:
                    raise ValueError(f"Parameter '{k}' has no initial value.")
            self._manually_queued_candidates.append(added_candidate)

    def _fill_missing_init_values(self):
        for k, v in self._params.items():
            if isinstance(v, Parameter) and self._best_solution[k] is None:
                self._best_solution[k] = v.sample()

    def sample_solution(self):
        candidate = {}
        for (k, v) in self._params.items():
            if isinstance(v, Parameter):
                candidate[k] = v.sample()
            else:
                candidate[k] = v
        return candidate

    def mutate_from_best(self, temperature):
        temperature = np.clip(temperature, 0, 1)
        candidate = {}
        for (k, v) in self._params.items():
            candidate[k] = self._best_solution[k]

        # With decreasing temperature we resample/mutate fewer parameters
        amount_to_mutate = int(
            max(round(temperature * len(self._free_params)), 1)
        )  # at least 1, at most all
        params_to_mutate = np.random.default_rng().choice(
            list(self._free_params.keys()), size=amount_to_mutate, replace=False
        )
        for k in params_to_mutate:
            candidate[k] = self._params[k].mutate(candidate[k], temperature)
        return candidate

    def _submit_candidate(self, objective_function, candidate_type, candidate, kwargs):
        for c in self._hooked_callbacks:
            c.on_evaluate_start(candidate)
        self._f_cache.stage(candidate)
        if self._task_executor is None:
            start = time.time()
            candidate_result = execute(
                objective_function,
                candidate,
                self._canceller,
                kwargs,
            )
            self._async_result_ready(
                candidate_type, candidate, time.time() - start, candidate_result
            )
        else:
            self._task_executor.submit(
                objective_function, candidate_type, candidate, self._canceller, kwargs
            )

    def _wait_for_one_free_executor(self):
        if self._task_executor is not None:
            if self._task_executor.is_full:
                # Queue is full, let's wait at least until 1 task is done before submitting this one
                self._task_executor.wait_for_first_to_complete()

            for (
                candidate_type,
                candidate,
                runtime,
                candidate_result,
            ) in self._task_executor.iterate_done_tasks():
                self._async_result_ready(
                    candidate_type, candidate, runtime, candidate_result
                )

    def _wait_for_all_running_jobs(self):
        if self._task_executor is not None:
            self._task_executor.wait_for_all_to_complete()
            for (
                candidate_type,
                candidate,
                runtime,
                candidate_result,
            ) in self._task_executor.iterate_done_tasks():
                self._async_result_ready(
                    candidate_type, candidate, runtime, candidate_result
                )

    def _async_result_ready(self, candidate_type, candidate, runtime, candidate_result):
        if candidate_result.cancelled_by_nan and not self._ignore_nans:
            raise ValueError(
                "NaN returned in objective function. If NaNs should be ignored (treated as cancelled evaluations) pass 'ignore_nans=True' argument to 'run'"
            )
        self._f_cache.commit(candidate, candidate_result.value)
        if (
            self._canceller is not None
            and not candidate_result.cancelled_by_user
            and not candidate_result.cancelled_by_nan
        ):
            # If the cancellation was done by the user or NaN we should not tell the EarlyCanceller object
            if candidate_result.intermediate_results is None:
                raise ValueError(
                    "An EarlyCanceller was passed but the objective function is not a generator"
                )
            self._canceller.append(candidate_result.intermediate_results)

        if candidate_result.was_cancelled:
            self._history.append_cancelled(candidate, candidate_type, runtime)
            self._run_history.append_cancelled(candidate_type)
            for c in self._hooked_callbacks:
                c.on_evaluate_cancelled(candidate)
            return

        for c in self._hooked_callbacks:
            c.on_evaluate_end(candidate, candidate_result.value)

        if (
            self._best_f is None
            or (self._direction == "max" and candidate_result.value > self._best_f)
            or (self._direction == "min" and candidate_result.value < self._best_f)
        ):
            # new best solution
            self._best_solution = candidate
            self._best_f = candidate_result.value
            for c in self._hooked_callbacks:
                c.on_new_best(self._best_solution, self._best_f)
        self._history.append(
            candidate, candidate_type, runtime, candidate_result.value, self._best_f
        )
        self._run_history.append(candidate_type, runtime, candidate_result.value)

    def _has_f_or_args_changed(self, objective_function, kwargs):
        hash_code = id(objective_function)
        for k, v in kwargs.items():
            hash_code += id(k) + id(v)
        if hash_code != self._objective_hash:
            self._objective_hash = hash_code
            return True
        return False

    def pretty_print_results(self):
        text_value_quadtuple = [
            (
                "Initial solution ",
                self._run_history.best_per_type[CandidateType.INIT],
                self._run_history.amount_per_type[CandidateType.INIT],
                self._run_history.cancelled_per_type[CandidateType.INIT],
                self._run_history.runtime_per_type[CandidateType.INIT],
            )
        ]
        if self._run_history.amount_per_type[CandidateType.MANUALLY_ADDED] > 0:
            text_value_quadtuple.append(
                (
                    "Manually added ",
                    self._run_history.best_per_type[CandidateType.MANUALLY_ADDED],
                    self._run_history.amount_per_type[CandidateType.MANUALLY_ADDED],
                    self._run_history.cancelled_per_type[CandidateType.MANUALLY_ADDED],
                    self._run_history.runtime_per_type[CandidateType.MANUALLY_ADDED],
                )
            )
        if self._run_history.amount_per_type[CandidateType.RANDOM_SEEDING] > 0:
            text_value_quadtuple.append(
                (
                    "Random seeding",
                    self._run_history.best_per_type[CandidateType.RANDOM_SEEDING],
                    self._run_history.amount_per_type[CandidateType.RANDOM_SEEDING],
                    self._run_history.cancelled_per_type[CandidateType.RANDOM_SEEDING],
                    self._run_history.runtime_per_type[CandidateType.RANDOM_SEEDING],
                )
            )
        if self._run_history.amount_per_type[CandidateType.LOCAL_SAMPLING] > 0:
            text_value_quadtuple.append(
                (
                    "Local sampling",
                    self._run_history.best_per_type[CandidateType.LOCAL_SAMPLING],
                    self._run_history.amount_per_type[CandidateType.LOCAL_SAMPLING],
                    self._run_history.cancelled_per_type[CandidateType.LOCAL_SAMPLING],
                    self._run_history.runtime_per_type[CandidateType.LOCAL_SAMPLING],
                )
            )
        text_value_quadtuple.append(
            (
                "Total",
                self._run_history.best_f,
                self._run_history.total_amount,
                self._run_history.total_cancelled,
                self._run_history.total_runtime,
            )
        )
        text_list = []
        for text, f, steps, cancelled, elapsed in text_value_quadtuple:
            value = "x" if f is None else f"{f:0.3g}"
            text_list.append(
                [
                    text,
                    value,
                    steps_to_pretty_str(steps),
                    steps_to_pretty_str(cancelled),
                    time_to_pretty_str(elapsed),
                ]
            )
        text_list.insert(0, ["Mode", "Best f", "Steps", "Cancelled", "Time"])
        text_list.insert(1, ["-----------", "---", "---", "---", "---"])
        text_list.insert(-1, ["-----------", "---", "---", "---", "---"])
        if self._run_history.total_cancelled == 0:
            # No candidate was cancelled so let's not show this column
            for t in text_list:
                t.pop(3)
        num_items = len(text_list[0])
        maxes = [
            np.max([len(text_list[j][i]) for j in range(len(text_list))])
            for i in range(num_items)
        ]
        line_len = np.sum(maxes) + 3 * (num_items - 1)
        line = ""
        for i in range(line_len // 2 - 4):
            line += "="
        line += " Summary "
        for i in range(line_len - len(line)):
            line += "="
        print(line)
        for j in range(len(text_list)):
            line = ""
            for i in range(num_items):
                if i > 0:
                    line += " : "
                line += text_list[j][i].ljust(maxes[i])
            print(line)
        line = ""
        for i in range(line_len):
            line += "="
        print(line)

    def _initialize_for_new_run(
        self, objective_function, direction, kwargs, canceller, ignore_nans
    ):
        if direction not in [
            "maximize",
            "maximise",
            "max",
            "minimize",
            "minimise",
            "min",
        ]:
            raise ValueError(
                f"Unknown direction '{direction}', must bei either 'maximize' or 'minimize'"
            )
        direction = direction.lower()[0:3]  # only save first 3 chars
        has_direction_changed = direction != self._direction
        self._direction = direction
        self._canceller = canceller
        if self._canceller is not None:
            self._canceller.direction = self._direction
        self._run_history = LocalHistory(self._direction)
        self._ignore_nans = ignore_nans

        self._fill_missing_init_values()
        if (
            self._has_f_or_args_changed(objective_function, kwargs)
            or has_direction_changed
        ):
            # not a single solution evaluated yet
            self._best_f = None  # Delete current best solution objective function value
            self._f_cache.clear()  # Reset cache
        elif self._best_f is not None:
            # not first run and objective function stayed the same
            self._run_history.hot_start(self._best_f)

    def _force_termination(self):
        # This is actually not needed but let's keep it for potential future use
        if self._task_executor is not None:
            self._task_executor.shutdown()

    def _hook_callbacks(self, callbacks):
        if callbacks is None:
            return
        if not isinstance(callbacks, list):
            # Convert single callback object to a list of size 1
            callbacks = [callbacks]
        self._hooked_callbacks.extend(callbacks)
        for c in self._hooked_callbacks:
            c.on_search_start(self)

    def _unhook_callbacks(self):
        for c in self._hooked_callbacks:
            c.on_search_end(self._history)
        self._hooked_callbacks.clear()

    def run(
        self,
        objective_function,
        direction: str = "maximize",
        timeout: Union[int, float, str, None] = None,
        max_steps: Optional[int] = None,
        seeding_steps: Optional[int] = None,
        seeding_timeout: Union[int, float, str, None] = None,
        seeding_ratio: Optional[float] = 0.3,
        canceller=None,
        n_jobs=1,
        quiet=False,
        ignore_nans=False,
        mp_backend="auto",
        enable_rejection_cache=True,
        callbacks: Union[callable, list, None] = None,
        start_temperature: float = 1,
        end_temperature: float = 0,
        kwargs=None,
    ):
        """
        :param direction: String defining if the objective function should be minimized or maximize
            (admissible values are 'min','minimize', or 'max','maximize')
        """
        if kwargs is None:
            kwargs = {}

        if len(self._free_params) == 0:
            raise ValueError(
                "There are not parameters to optimize (search space does not contain any `pyhopper.Parameter` instance)"
            )

        self._current_run_config = {
            "direction": direction,
            "timeout": timeout,
            "max_steps": max_steps,
            "seeding_steps": seeding_steps,
            "seeding_timeout": seeding_timeout,
            "seeding_ratio": seeding_ratio,
            "n_jobs": n_jobs,
            "use_canceller": canceller is not None,
            "ignore_nans": ignore_nans,
            "start_temperature": start_temperature,
            "end_temperature": end_temperature,
        }
        schedule = ScheduledRun(
            max_steps,
            timeout,
            seeding_steps=seeding_steps,
            seeding_timeout=seeding_timeout,
            seeding_ratio=seeding_ratio,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
        )
        self._f_cache.set_enable(enable_rejection_cache)
        self._signal_listener.register_signal(
            schedule.signal_gradually_quit, self._force_termination
        )
        if not quiet:
            print(f"Search is scheduled for {schedule.to_total_str()}")
        if n_jobs != 1:
            self._task_executor = TaskManager(n_jobs, mp_backend)
            if self._task_executor.n_jobs == 1:
                self._task_executor = None  # '1x per-gpu' on single GPU machines -> No need for multiprocess overhead

        self._hook_callbacks(callbacks)

        self._initialize_for_new_run(
            objective_function, direction, kwargs, canceller, ignore_nans
        )
        pbar = ProgBar(schedule, self._run_history, disable=quiet)
        if self._best_f is None:
            # Evaluate initial guess, this gives the user some estimate of how much PyHopper could tune the parameters
            self._submit_candidate(
                objective_function,
                CandidateType.INIT,
                self._best_solution,
                kwargs,
            )
            schedule.increment_step()

        current_temperature = schedule.temperature
        # Before entering the loop, let's wait until we can run at least one candidate
        self._wait_for_one_free_executor()
        while not schedule.is_timeout(self._run_history.estimated_candidate_runtime):
            # If estimated runtime exceeds timeout let's already terminate
            if len(self._manually_queued_candidates) > 0:
                candidate = self._manually_queued_candidates.pop(-1)
                candidate_type = CandidateType.MANUALLY_ADDED
            elif (
                schedule.is_mixed_endless
                and np.random.default_rng().random() < schedule.endless_seeding_ratio
            ) or (
                not schedule.is_seeding_timeout(
                    self._run_history.estimated_candidate_runtime
                )
            ):
                candidate = self.sample_solution()
                candidate_type = CandidateType.RANDOM_SEEDING
            else:
                candidate = self.mutate_from_best(temperature=current_temperature)
                candidate_type = CandidateType.LOCAL_SAMPLING
            if candidate not in self._f_cache:
                # If candidate was already run before, let's skip this step
                self._submit_candidate(
                    objective_function,
                    candidate_type,
                    candidate,
                    kwargs,
                )
                current_temperature = schedule.temperature
            else:
                # Reject sample
                current_temperature *= (
                    1.05  # increase temperature by 5% if we found a duplicate
                )
                current_temperature = max(current_temperature, 1)
            schedule.increment_step()
            pbar.update(self._best_f)
            # Before entering the loop, let's wait until we can run at least one candidate
            self._wait_for_one_free_executor()

        self._wait_for_all_running_jobs()

        self._unhook_callbacks()
        self._signal_listener.unregister_signal()
        pbar.close(self._best_f)

        # Clean up the task executor
        del self._task_executor
        self._task_executor = None

        if not quiet:
            self.pretty_print_results()
        return self._best_solution

    @property
    def current_run_config(self):
        return self._current_run_config

    @property
    def history(self):
        return self._history

    @property
    def best(self):
        return self._best_solution

    @property
    def best_f(self):
        return self._best_f