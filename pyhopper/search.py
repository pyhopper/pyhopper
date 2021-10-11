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
from typing import Union, Optional, Any, Tuple, Sequence
from types import FunctionType
from enum import Enum
import time

from .run_context import ScheduledRun, RunContext
from .utils import (
    parse_timeout,
    sanitize_bounds,
    infer_shape,
    time_to_pretty_str,
    steps_to_pretty_str,
    ParamInfo,
    CandidateType,
    merge_dicts,
)


def register_int(
    lb: Optional[Union[int, float, np.ndarray]] = None,
    ub: Optional[Union[int, float, np.ndarray]] = None,
    init: Optional[Union[int, float, np.ndarray]] = None,
    multiple_of: Optional[int] = None,
    power_of: Optional[int] = None,
    shape: Optional[Union[int, Tuple]] = None,
    seeding_fn: Optional[callable] = None,
    mutation_fn: Optional[callable] = None,
) -> IntParameter:
    """Creates a new integer parameter

    :param lb: Lower bound of the parameter.
    :param ub: Upper bound of the parameter. If None, the `lb` argument will be used as upper bound with a lower bound of 0.
    :param init: Initial value of the parameter. If None it will be randomly sampled
    :param multiple_of: Setting this value to a positive integer enforces the sampled values of this parameter to be a mulitple of `multiple_of`.
    :param shape: For NumPy array type parameters, this argument must be set to a tuple containing the shape of the np.ndarray
    :param mutation_fn: Setting this argument to a callable overwrites the default local sampling strategy. The callback gets called with the value
        of the the current best solution as argument and returns a mutated value
    :param seeding_fn: Setting this argument to a callable overwrites the default random seeding strategy
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
            mutation_fn,
            seeding_fn,
        )
    param = IntParameter(
        param_shape,
        lb,
        ub,
        init,
        multiple_of,
        mutation_fn,
        seeding_fn,
    )
    return param


def register_custom(
    seeding_fn: Optional[callable] = None,
    mutation_fn: Optional[callable] = None,
    init: Any = None,
) -> CustomParameter:
    if seeding_fn is None and init is None:
        raise ValueError(
            f"Could not create custom parameter, must either provide an initial value or a seeding strategy function"
        )
    param = CustomParameter(init, mutation_fn, seeding_fn)
    return param


def register_choice(
    options: list,
    init: Optional[Any] = None,
    is_ordinal: bool = False,
    mutation_fn: Optional[FunctionType] = None,
    seeding_fn: Optional[FunctionType] = None,
) -> ChoiceParameter:
    """Creates a new choice parameter

    :param options: List containing the possible values of this parameter
    :param init: Initial value of the parameter. If None it will be randomly sampled.
    :param is_ordinal: Flag indicating whether two neighboring list items ordered or not. If True, in the local sampling stage list items neighboring the current best value will be preferred. For sets with a natural ordering it is recommended to set this flag to True.
    :param mutation_fn: Setting this argument to a callable overwrites the default local sampling strategy. The callback gets called with the value
        of the the current best solution as argument and returns a mutated value
    :param seeding_fn: Setting this argument to a callable overwrites the default random seeding strategy
    :return:
    """
    if len(options) == 0:
        raise ValueError("List with possible values must not be empty.")
    param = ChoiceParameter(options, init, is_ordinal, mutation_fn, seeding_fn)
    return param


def register_float(
    lb: Optional[Union[int, float, np.ndarray]] = None,
    ub: Optional[Union[int, float, np.ndarray]] = None,
    init: Optional[Union[int, float, np.ndarray]] = None,
    log: Union[bool] = False,
    precision: Optional[int] = None,
    shape: Optional[Union[int, Tuple]] = None,
    mutation_fn: Optional[FunctionType] = None,
    seeding_fn: Optional[FunctionType] = None,
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
    :param mutation_fn: Setting this argument to a callable overwrites the default local sampling strategy. The callback gets called with the value
        of the the current best solution as argument and returns a mutated value
    :param seeding_fn: Setting this argument to a callable overwrites the default random seeding strategy
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
            param_shape, lb, ub, init, precision, mutation_fn, seeding_fn
        )
    param = FloatParameter(
        param_shape,
        lb,
        ub,
        init,
        precision,
        mutation_fn,
        seeding_fn,
    )
    return param


class Search:
    def __init__(self, *args: Union[dict, Sequence[dict]], **kwargs):
        """
        Creates a new search object

        :param args: dict defining the search space. If multiple dicts are provided the dicts will be merged.
        param kwargs: key-value pairs defining the search space. Will be merged with the numbered arguments if some are provided
        """
        parameters = {}
        if len(args) > 0:
            parameters = merge_dicts(*args)

        parameters = merge_dicts(parameters, kwargs)

        self._params = {}
        self._best_solution = {}
        self._free_params = {}
        for k, v in parameters.items():
            self._register_parameter(k, v)
        self._best_f = None
        self._f_cache = EvaluationCache()
        self._run_context = None
        self._manually_queued_candidates = []

        self._signal_listener = SignalListener()

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
        param_info = ParamInfo(candidate_type, sampled_at=time.time())
        for c in self._run_context.callbacks:
            c.on_evaluate_start(candidate, param_info)

        self._f_cache.stage(candidate)
        if self._run_context.task_executor is None:
            candidate_result = execute(
                objective_function,
                candidate,
                self._run_context.canceler,
                kwargs,
            )
            param_info.finished_at = time.time()
            self._async_result_ready(candidate, param_info, candidate_result)
        else:
            self._run_context.task_executor.submit(
                objective_function,
                candidate,
                param_info,
                self._run_context.canceler,
                kwargs,
            )

    def _wait_for_one_free_executor(self):
        if self._run_context.task_executor is not None:
            if self._run_context.task_executor.is_full:
                # Queue is full, let's wait at least until 1 task is done before submitting this one
                self._run_context.task_executor.wait_for_first_to_complete()

            for (
                candidate,
                param_info,
                candidate_result,
            ) in self._run_context.task_executor.iterate_done_tasks():
                self._async_result_ready(candidate, param_info, candidate_result)

    def _wait_for_all_running_jobs(self):
        if self._run_context.task_executor is not None:
            self._run_context.task_executor.wait_for_all_to_complete()
            for (
                candidate,
                param_info,
                candidate_result,
            ) in self._run_context.task_executor.iterate_done_tasks():
                self._async_result_ready(candidate, param_info, candidate_result)

    def _async_result_ready(self, candidate, param_info, candidate_result):
        if candidate_result.canceled_by_nan and not self._run_context.ignore_nans:
            raise ValueError(
                "NaN returned in objective function. If NaNs should be ignored (treated as canceled evaluations) pass 'ignore_nans=True' argument to 'run'"
            )
        self._f_cache.commit(candidate, candidate_result.value)
        if (
            self._run_context.canceler is not None
            and not candidate_result.canceled_by_user
            and not candidate_result.canceled_by_nan
        ):
            # If the cancelation was done by the user or NaN we should not tell the Earlycanceler object
            if candidate_result.intermediate_results is None:
                raise ValueError(
                    "An Earlycanceler was passed but the objective function is not a generator"
                )
            self._run_context.canceler.append(candidate_result.intermediate_results)

        if candidate_result.was_canceled:
            param_info.is_canceled = True
            for c in self._run_context.callbacks:
                c.on_evaluate_canceled(candidate, param_info)
            return

        for c in self._run_context.callbacks:
            c.on_evaluate_end(candidate, candidate_result.value, param_info)

        if (
            self._best_f is None
            or (
                self._run_context.direction == "max"
                and candidate_result.value > self._best_f
            )
            or (
                self._run_context.direction == "min"
                and candidate_result.value < self._best_f
            )
        ):
            # new best solution
            self._best_solution = candidate
            self._best_f = candidate_result.value
            for c in self._run_context.callbacks:
                c.on_new_best(self._best_solution, self._best_f, param_info)

    def _force_termination(self):
        # This is actually not needed but let's keep it for potential future use
        if self._run_context.task_executor is not None:
            self._run_context.task_executor.shutdown()

    def run(
        self,
        objective_function,
        direction: str = "maximize",
        timeout: Union[int, float, str, None] = None,
        max_steps: Optional[int] = None,
        seeding_steps: Optional[int] = None,
        seeding_timeout: Union[int, float, str, None] = None,
        seeding_ratio: Optional[float] = 0.3,
        canceler=None,
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

        schedule = ScheduledRun(
            max_steps,
            timeout,
            seeding_steps=seeding_steps,
            seeding_timeout=seeding_timeout,
            seeding_ratio=seeding_ratio,
            start_temperature=start_temperature,
            end_temperature=end_temperature,
        )
        task_executor = None
        if n_jobs != 1:
            task_executor = TaskManager(n_jobs, mp_backend)
            if task_executor.n_jobs == 1:
                task_executor = None  # '1x per-gpu' on single GPU machines -> No need for multiprocess overhead

        self._run_context = RunContext(
            direction,
            canceler,
            ignore_nans,
            schedule,
            callbacks,
            task_executor,
            quiet,
        )

        self._current_run_config = {
            "direction": direction,
            "timeout": timeout,
            "max_steps": max_steps,
            "seeding_steps": seeding_steps,
            "seeding_timeout": seeding_timeout,
            "seeding_ratio": seeding_ratio,
            "n_jobs": n_jobs,
            "use_canceler": canceler is not None,
            "ignore_nans": ignore_nans,
            "start_temperature": start_temperature,
            "end_temperature": end_temperature,
        }
        self._f_cache.set_enable(enable_rejection_cache)
        self._signal_listener.register_signal(
            schedule.signal_gradually_quit, self._force_termination
        )
        self._fill_missing_init_values()
        for c in self._run_context.callbacks:
            c.on_search_start(self)

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
        while not schedule.is_timeout(
            self._run_context.run_history.estimated_candidate_runtime
        ):
            # If estimated runtime exceeds timeout let's already terminate
            if len(self._manually_queued_candidates) > 0:
                candidate = self._manually_queued_candidates.pop(-1)
                candidate_type = CandidateType.MANUALLY_ADDED
            elif (
                schedule.is_mixed_endless
                and np.random.default_rng().random() < schedule.endless_seeding_ratio
            ) or (
                not schedule.is_seeding_timeout(
                    self._run_context.run_history.estimated_candidate_runtime
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
            # Before entering the loop, let's wait until we can run at least one candidate
            self._wait_for_one_free_executor()

        self._wait_for_all_running_jobs()

        for c in self._run_context.callbacks:
            c.on_search_end()
        self._signal_listener.unregister_signal()

        # Clean up the run context (task executor,progbar,run history)
        del self._run_context
        self._run_context = None

        return self._best_solution

    @property
    def current_run_config(self):
        return self._current_run_config

    @property
    def best(self):
        return self._best_solution

    @property
    def best_f(self):
        return self._best_f