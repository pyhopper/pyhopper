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

import os

from .cache import EvaluationCache
from .parameters import (
    FloatParameter,
    IntParameter,
    ChoiceParameter,
    CustomParameter,
    FrozenParameter,
    FixedParameter,
    Parameter,
)
from .parallel import execute, TaskManager, EvaluationResult, SignalListener
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
    NEIGHBORHOOD_SAMPLING = 3


class ScheduledRun:
    def __init__(
        self,
        max_steps: Optional[int] = None,
        timeout: Union[int, float, str, None] = None,
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
        self._sigterm_received = 0

        self._temp_start = None
        self._force_quit_callback = None
        self._original_sigint_handler = None
        self.reset_temperature()

    def signal_gradually_quit(self):
        self._sigterm_received += 1

    def increment_step(self):
        self._step += 1

    @property
    def step(self):
        return self._step

    @property
    def current_runtime(self):
        return time.time() - self._start_time

    @property
    def is_endless(self):
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
    def is_disabled(self):
        if self._step_limit is not None:
            # step-scheduled mode
            return self._step_limit <= 0
        elif self._timeout is not None:
            # time-scheduled mode
            return self._timeout <= 0
        return False

    def is_timeout(self):
        if self._sigterm_received > 0:
            return True
        if self._step_limit is not None:
            # step-scheduled mode
            return self._step >= self._step_limit
        elif self._timeout is not None:
            # time-scheduled mode
            return time.time() - self._start_time >= self._timeout
        else:
            return False

    def reset_temperature(self):
        if self._step_limit is not None:
            self._temp_start = self._step
        else:
            self._temp_start = time.time() - self._start_time

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
        if self._step_limit is not None:
            # step-scheduled mode
            progress = (self._step - self._temp_start) / (
                self._step_limit - self._temp_start
            )
            return 1.0 - progress
        else:
            # time-scheduled mode
            elapsed = time.time() - self._start_time
            progress = (elapsed - self._temp_start) / (self._timeout - self._temp_start)
            return 1.0 - progress


class History:
    def __init__(self, direction):
        self._direction = direction
        self._start_time = time.time()
        self._log_candidate_types = []
        self._log_candidate_f = []
        self._log_candidate_time = []
        self._log_candidate_best_f = []

        self._best_per_type = {
            CandidateType.INIT: None,
            CandidateType.MANUALLY_ADDED: None,
            CandidateType.RANDOM_SEEDING: None,
            CandidateType.NEIGHBORHOOD_SAMPLING: None,
        }
        self._amount_per_type = {
            CandidateType.INIT: 0,
            CandidateType.MANUALLY_ADDED: 0,
            CandidateType.RANDOM_SEEDING: 0,
            CandidateType.NEIGHBORHOOD_SAMPLING: 0,
        }
        self._cancelled_per_type = {
            CandidateType.INIT: 0,
            CandidateType.MANUALLY_ADDED: 0,
            CandidateType.RANDOM_SEEDING: 0,
            CandidateType.NEIGHBORHOOD_SAMPLING: 0,
        }
        self._total_cancelled = 0
        self._best_f = None

        self._init_time = None
        self._manually_time = None
        self._seeding_time = None
        self._neighbor_time = None

    def is_better(self, old, new):
        return (
            old is None
            or (self._direction == "max" and new > old)
            or (self._direction == "min" and new < old)
        )

    def candidate_cancelled(self, candidate_type):
        self._cancelled_per_type[candidate_type] += 1
        self._total_cancelled += 1

    def hot_start(self, best_f):
        self._best_f = best_f
        self._best_per_type[CandidateType.INIT] = best_f

    def append(self, candidate_type, f):
        self._log_candidate_types.append(candidate_type)
        self._log_candidate_f.append(f)
        self._log_candidate_time.append(time.time() - self._start_time)
        if self.is_better(self._best_f, f):
            self._best_f = f
        self._log_candidate_best_f.append(self._best_f)
        if self.is_better(self._best_per_type[candidate_type], f):
            self._best_per_type[candidate_type] = f
        self._amount_per_type[candidate_type] += 1

    def as_dict(self):
        return {
            "best_f": self._log_candidate_best_f,
            "candidate_f": self._log_candidate_f,
            "candidate_time": self._log_candidate_time,
            "candidate_type": self._log_candidate_types,
        }

    def start_manually_added_phase(self):
        self._init_time = time.time() - self._start_time

    def start_seeding_phase(self):
        self._manually_time = time.time() - self._start_time - self._init_time

    def start_neighbor_phase(self):
        self._seeding_time = (
            time.time() - self._start_time - self._manually_time - self._init_time
        )

    def start_endless_phase(self):
        self._manually_time = time.time() - self._start_time - self._init_time

    def end_endless_phase(self, ratio):
        elapsed = time.time() - self._start_time - self._manually_time - self._init_time
        self._seeding_time = (1.0 - ratio) * elapsed
        self._neighbor_time = ratio * elapsed
        self._total_time = time.time() - self._start_time

    def end_neighbor_phase(self):
        self._neighbor_time = (
            time.time()
            - self._start_time
            - self._manually_time
            - self._init_time
            - self._seeding_time
        )
        self._total_time = time.time() - self._start_time


def register_int(
    lb: Optional[Union[int, float, np.ndarray]] = None,
    ub: Optional[Union[int, float, np.ndarray]] = None,
    init: Optional[Union[int, float, np.ndarray]] = None,
    multiple_of: Optional[int] = None,
    shape: Optional[Tuple] = None,
    mutation_strategy: Optional[FunctionType] = None,
    sampling_strategy: Optional[FunctionType] = None,
) -> IntParameter:
    if lb is None and ub is None:
        # Unbounded int is actually a 32-bit integer
        lb = -(2 ** 31)
        ub = 2 ** 31 - 1
    lb, ub = sanitize_bounds(lb, ub)
    param_shape = infer_shape(shape, init, lb, ub) if shape is None else shape
    param = IntParameter(
        param_shape, lb, ub, init, multiple_of, mutation_strategy, sampling_strategy
    )
    return param


def register_custom(
    init: Any,
    mutation_strategy: Optional[FunctionType] = None,
    sampling_strategy: Optional[FunctionType] = None,
) -> CustomParameter:
    if sampling_strategy is None and init is None:
        raise ValueError(
            f"Could not create custom parameter, must either provide an initial value or a seeding strategy function"
        )
    param = CustomParameter(init, mutation_strategy, sampling_strategy)
    return param


def register_choice(
    options: list,
    init: Optional[Any] = None,
    is_ordinal: bool = False,
    mutation_strategy: Optional[FunctionType] = None,
    sampling_strategy: Optional[FunctionType] = None,
) -> ChoiceParameter:
    param = ChoiceParameter(
        options, init, is_ordinal, mutation_strategy, sampling_strategy
    )
    return param


def register_float(
    lb: Optional[Union[int, float, np.ndarray]] = None,
    ub: Optional[Union[int, float, np.ndarray]] = None,
    init: Optional[Union[int, float, np.ndarray]] = None,
    log: Union[bool] = False,
    precision: Optional[int] = None,
    shape: Optional[Tuple] = None,
    mutation_strategy: Optional[FunctionType] = None,
    sampling_strategy: Optional[FunctionType] = None,
) -> FloatParameter:
    """

    :param lb: Lower bound of the parameter.
    :param ub: Upper bound of the parameter
    :param init: Initial value of the parameter. If None it will be sampled from the interval
    :param log: Whether to use logarithmic or uniform scaling of the parameter.
        Defaults to False which searches the space uniformly.
        If True, a logarithmic scaling is applied to the search space of this variable
    :param precision: Rounds the values to the specified significant digits.
        Defaults to None meaning that no rounding is applied
    :param mutation_strategy: Manual mutations can be implemented via a callback that maps a value
        of the the current best solution to a mutated value
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
    param = FloatParameter(
        param_shape,
        lb,
        ub,
        init,
        log,
        precision,
        mutation_strategy,
        sampling_strategy,
    )
    return param


class Search:
    def __init__(
        self,
        parameters: dict,
        direction: str = "maximize",
        on_new_best_callback: FunctionType = None,
        on_candidate_evaluated_callback: FunctionType = None,
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
        self._direction = direction.lower()[0:3]  # only save first 3 chars

        self._params = {}
        self._best_solution = {}
        self._best_f = None
        self._on_new_best_callback = on_new_best_callback
        self._on_candidate_evaluated_callback = on_candidate_evaluated_callback
        self._canceller = None
        self._ignore_nans = None

        self._history = None
        self._f_cache = EvaluationCache()
        self._reported_history = None
        self._manually_queued_candidates = []

        self._signal_listener = SignalListener()
        self._schedule = []
        self._objective_hash = None
        self._task_executor = None

        for k, v in parameters.items():
            self._register_parameter(k, v)

    def enqueue_candidate(self, candidate: dict) -> None:
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

    def enqueue_parameter_sweep(self, name: str, candidate_values: list) -> None:
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

    def freeze_parameter(self, name: str):
        if name not in self._params.keys():
            raise ValueError(f"Parameter with name '{name}' does not exist")
        if self._best_solution[name] is None:
            raise ValueError(
                f"Cannot freeze parameter '{name}' because it has no initial value!"
            )
        old_param = self._params.get(name)
        self._params[name] = FrozenParameter(self._best_solution[name], old_param)

    def unfreeze_parameter(self, name: str):
        if name not in self._params.keys():
            raise ValueError(f"Parameter with name '{name}' does not exists")
        param = self._params[name]
        if not isinstance(param, FrozenParameter):
            raise ValueError(f"Parameter '{name}' is not frozen")
        restored_param = param.get_old()
        self._params[name] = restored_param

    def _register_parameter(self, name: str, param: Any) -> None:
        if name in self._params.keys():
            raise ValueError(f"Parameter with name '{name}' already exists")
        if isinstance(param, Parameter):
            self._params[name] = param
            self._best_solution[name] = param._init
        else:
            self._params[name] = FixedParameter(param)
            self._best_solution[name] = param

    def _fill_missing_init_values(self):
        for k, v in self._params.items():
            if self._best_solution[k] is None:
                self._best_solution[k] = v.sample()

    def sample_solution(self):
        candidate = {}
        for (k, v) in self._params.items():
            candidate[k] = v.sample()
        return candidate

    def mutate_from_best(self, temperature):
        temperature = np.clip(temperature, 0, 1)
        candidate = {}
        for (k, v) in self._params.items():
            candidate[k] = v.mutate(self._best_solution[k], temperature)
        return candidate

    def _submit_candidate(self, objective_function, candidate_type, candidate, kwargs):
        self._f_cache.stage(candidate)
        if self._task_executor is None:
            candidate_result = execute(
                objective_function, candidate, self._canceller, kwargs
            )
            self._async_result_ready(candidate_type, candidate, candidate_result)
        else:
            # if self._task_executor.is_full:
            #     # Queue is full, let's wait at least until 1 task is done before submitting this one
            #     self._task_executor.wait_for_first_to_complete()
            self._task_executor.submit(
                objective_function, candidate_type, candidate, self._canceller, kwargs
            )
            # # Let's collect all tasks that are complete
            # for (
            #     candidate_type,
            #     candidate,
            #     candidate_f,
            # ) in self._task_executor.iterate_done_tasks():
            #     self._async_result_ready(candidate_type, candidate, candidate_f)

    def _wait_for_free_executor(self):
        if self._task_executor is not None:
            if self._task_executor.is_full:
                # Queue is full, let's wait at least until 1 task is done before submitting this one
                self._task_executor.wait_for_first_to_complete()

            for (
                candidate_type,
                candidate,
                candidate_result,
            ) in self._task_executor.iterate_done_tasks():
                self._async_result_ready(candidate_type, candidate, candidate_result)

    def _wait_for_running_jobs(self):
        if self._task_executor is not None:
            self._task_executor.wait_for_all_to_complete()
            for (
                candidate_type,
                candidate,
                candidate_result,
            ) in self._task_executor.iterate_done_tasks():
                self._async_result_ready(candidate_type, candidate, candidate_result)

    def _async_result_ready(self, candidate_type, candidate, candidate_result):
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
            self._history.candidate_cancelled(candidate_type)
            return

        if self._on_candidate_evaluated_callback is not None:
            self._on_candidate_evaluated_callback(candidate, candidate_result.value)

        if (
            self._best_f is None
            or (self._direction == "max" and candidate_result.value > self._best_f)
            or (self._direction == "min" and candidate_result.value < self._best_f)
        ):
            # new best solution
            self._best_solution = candidate
            self._best_f = candidate_result.value
            if self._on_new_best_callback is not None:
                self._on_new_best_callback(self._best_solution, self._best_f)

        self._history.append(candidate_type, candidate_result.value)

    # def _evaluate_enqueued_candidates(self, objective_function, kwargs, schedule=None):
    #     while len(self._manually_queued_candidates) > 0:
    #         self._wait_for_free_executor()
    #         if schedule is not None and schedule.is_timeout():
    #             break
    #         candidate = self._manually_queued_candidates.pop(-1)
    #         self._submit_candidate(
    #             objective_function, CandidateType.MANUALLY_ADDED, candidate, kwargs
    #         )

    def _has_changed(self, objective_function, kwargs):
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
                self._history._best_per_type[CandidateType.INIT],
                self._history._amount_per_type[CandidateType.INIT],
                self._history._cancelled_per_type[CandidateType.INIT],
                self._history._init_time,
            )
        ]
        if self._history._amount_per_type[CandidateType.MANUALLY_ADDED] > 0:
            text_value_quadtuple.append(
                (
                    "Manually added ",
                    self._history._best_per_type[CandidateType.MANUALLY_ADDED],
                    self._history._amount_per_type[CandidateType.MANUALLY_ADDED],
                    self._history._cancelled_per_type[CandidateType.MANUALLY_ADDED],
                    self._history._manually_time,
                )
            )
        if self._history._amount_per_type[CandidateType.RANDOM_SEEDING] > 0:
            text_value_quadtuple.append(
                (
                    "Random seeding",
                    self._history._best_per_type[CandidateType.RANDOM_SEEDING],
                    self._history._amount_per_type[CandidateType.RANDOM_SEEDING],
                    self._history._cancelled_per_type[CandidateType.RANDOM_SEEDING],
                    self._history._seeding_time,
                )
            )
        if self._history._amount_per_type[CandidateType.NEIGHBORHOOD_SAMPLING] > 0:
            text_value_quadtuple.append(
                (
                    "Neighborhood sampling",
                    self._history._best_per_type[CandidateType.NEIGHBORHOOD_SAMPLING],
                    self._history._amount_per_type[CandidateType.NEIGHBORHOOD_SAMPLING],
                    self._history._cancelled_per_type[
                        CandidateType.NEIGHBORHOOD_SAMPLING
                    ],
                    self._history._neighbor_time,
                )
            )
        text_value_quadtuple.append(
            (
                "Total",
                self._history._best_f,
                len(self._history._log_candidate_f),
                self._history._total_cancelled,
                self._history._total_time,
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
        if self._history._total_cancelled == 0:
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

    def _update_pbar(self, pbar):
        if self._best_f is None:
            pbar.set_description(f"Currently running")
        else:
            pbar.set_description(f"Current best {self._best_f:0.3g}")

    def _initialize_for_new_run(
        self, objective_function, kwargs, canceller, ignore_nans
    ):
        self._canceller = canceller
        if self._canceller is not None:
            self._canceller.direction = self._direction
        self._history = History(self._direction)
        self._ignore_nans = ignore_nans

        if self._has_changed(objective_function, kwargs):
            # not a single solution evaluated yet
            self._best_f = None  # Delete current best solution objective function value
            self._f_cache.clear()  # Reset cache
            self._fill_missing_init_values()
            self._submit_candidate(
                objective_function, CandidateType.INIT, self._best_solution, kwargs
            )
        elif self._best_f is not None:
            # not first run and objective function stayed the same
            self._history.hot_start(self._best_f)

    def _save_best_as_pickle(self):
        print("SAVED")
        try:
            if self._task_executor is not None:
                self._task_executor.shutdown()
        except:
            pass
        # if current_runtime > 10 * 60 and self.best_f is not None:
        #     from datetime import datetime
        #     import pickle
        #
        #     os.makedirs("aborted_searches", exist_ok=True)
        #     fname = os.path.join(
        #         "aborted_searches", datetime.now().strftime("%Y%m%d_%H%M%S.pkl")
        #     )
        #     if not os.path.isfile(fname):
        #         with open(fname, "wb") as f:
        #             pickle.dump(self._best_solution, f)
        #         print(
        #             f"Current best candidate with {self.best_f:0.3g} saved in file '{fname}'"
        #         )
        #         print(
        #             f"Retrieve it with\n\nimport pickle\n\nwith open('{fname}','rb') as f:\n  restored = pickle.load(f)\n\n"
        #         )

    def _should_cache(self):
        # Method to check if the parameter space contains a numpy float array
        # if so, then the default setting is to not use a rejection cache as there
        # are only very few duplicates expected, which is not worth the hashing overhead
        has_float_np_array = False
        for (k, v) in self._params.items():
            if isinstance(v, FloatParameter):
                if v._shape is not None and len(v._shape) > 0:
                    has_float_np_array = True
        return not has_float_np_array

    def run(
        self,
        objective_function,
        max_steps: Optional[int] = None,
        timeout: Union[int, float, str, None] = None,
        seeding_steps: Optional[int] = None,
        seeding_timeout: Union[int, float, str, None] = None,
        canceller=None,
        n_jobs=1,
        verbose=1,
        ignore_nans=False,
        mp_backend="auto",
        enable_rejection_cache=None,
        kwargs=None,
    ):
        if kwargs is None:
            kwargs = {}

        seeding_schedule = ScheduledRun(seeding_steps, seeding_timeout)
        schedule = ScheduledRun(max_steps, timeout)
        if enable_rejection_cache is None:
            enable_rejection_cache = self._should_cache()
        self._f_cache.set_enable(enable_rejection_cache)
        self._signal_listener.register_signal(
            schedule.signal_gradually_quit, self._save_best_as_pickle
        )
        if verbose > 0:
            print(f"Search is scheduled for {schedule.to_total_str()}")
        if n_jobs != 1:
            self._task_executor = TaskManager(n_jobs, mp_backend)

        # import signal

        def handler(signum, frame):
            print("Signal handler called with signal", signum)
            # raise OSError("Couldn't open device!")

            # Set the signal handler and a 5-second alarm

        # signal.signal(signal.SIGINT, handler)

        pbar = tqdm(total=schedule.total_units, disable=verbose <= 0)

        self._initialize_for_new_run(objective_function, kwargs, canceller, ignore_nans)
        schedule.increment_step()

        pbar.n = round(schedule.current_units, 1)
        self._update_pbar(pbar)

        self._history.start_manually_added_phase()
        self._wait_for_free_executor()  #  Before entering the loop, let's wait until we can run at least one candidate
        while len(self._manually_queued_candidates) > 0 and not schedule.is_timeout():
            candidate = self._manually_queued_candidates.pop(-1)
            self._submit_candidate(
                objective_function, CandidateType.MANUALLY_ADDED, candidate, kwargs
            )
            pbar.n = round(schedule.current_units, 1)
            self._update_pbar(pbar)
            self._wait_for_free_executor()  # Before checking the loop condition wait for at least one job to finish

        if schedule.is_endless and seeding_schedule.is_endless:
            # In endless mode we do 80% neighborhood sampling and 20% random seeding
            self._history.start_endless_phase()
            self._wait_for_free_executor()  # Before entering the loop, let's wait until we can run at least one candidate
            while not schedule.is_timeout():  # Wait for SIG_INT (Ctrl+C)
                if np.random.default_rng().random() > 0.8:
                    seeding_schedule.increment_step()
                    schedule.increment_step()
                    candidate = self.sample_solution()
                    if candidate not in self._f_cache:
                        # If candidate was already run before, let's skip this step
                        self._submit_candidate(
                            objective_function,
                            CandidateType.RANDOM_SEEDING,
                            candidate,
                            kwargs,
                        )
                else:
                    candidate = self.mutate_from_best(
                        temperature=np.random.default_rng().uniform(0.1, 0.9)
                    )
                    if candidate not in self._f_cache:
                        # If candidate was already run before, let's skip this step
                        self._submit_candidate(
                            objective_function,
                            CandidateType.NEIGHBORHOOD_SAMPLING,
                            candidate,
                            kwargs,
                        )
                    schedule.increment_step()
                self._update_pbar(pbar)
                self._wait_for_free_executor()  # Before checking the loop condition wait for at least one job to finish

            self._wait_for_running_jobs()
            self._history.end_endless_phase(0.8)
        else:
            # In scheduled mode we first to random seeding and then improvement
            self._history.start_seeding_phase()
            self._wait_for_free_executor()  # Before entering the loop, let's wait until we can run at least one candidate
            while (
                not schedule.is_timeout()
                and not seeding_schedule.is_timeout()
                and not seeding_schedule.is_endless
            ):
                candidate = self.sample_solution()
                if candidate not in self._f_cache:
                    # If candidate was already run before, let's skip this step
                    self._submit_candidate(
                        objective_function,
                        CandidateType.RANDOM_SEEDING,
                        candidate,
                        kwargs,
                    )
                schedule.increment_step()
                seeding_schedule.increment_step()
                pbar.n = round(schedule.current_units, 1)
                self._update_pbar(pbar)
                self._wait_for_free_executor()  # Before checking the loop condition wait for at least one job to finish

            self._history.start_neighbor_phase()
            schedule.reset_temperature()
            self._wait_for_free_executor()  # Before entering the loop, let's wait until we can run at least one candidate

            current_temperature = schedule.temperature
            while not schedule.is_timeout():
                candidate = self.mutate_from_best(temperature=current_temperature)
                if candidate not in self._f_cache:
                    # If candidate was already run before, let's skip this step
                    self._submit_candidate(
                        objective_function,
                        CandidateType.NEIGHBORHOOD_SAMPLING,
                        candidate,
                        kwargs,
                    )
                    current_temperature = schedule.temperature
                else:
                    # Reject sample
                    current_temperature *= (
                        1.05  # increase temperature by 5% if we found a duplicate
                    )

                schedule.increment_step()
                pbar.n = round(schedule.current_units, 1)
                self._update_pbar(pbar)
                self._wait_for_free_executor()  # Before checking the loop condition wait for at least one job to finish

            self._wait_for_running_jobs()
            self._history.end_neighbor_phase()

        self._signal_listener.unregister_signal()
        pbar.n = schedule.total_units
        pbar.refresh()
        pbar.close()

        # Clean up the task executor
        del self._task_executor
        self._task_executor = None

        if verbose > 0:
            self.pretty_print_results()
        self._reported_history = self._history.as_dict()
        return self._best_solution

    @property
    def history(self):
        return self._reported_history

    @property
    def best(self):
        return self._best_solution

    @property
    def best_f(self):
        return self._best_f