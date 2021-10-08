from tqdm.auto import tqdm
import numpy as np
from pyhopper.callbacks import Callback
from pyhopper.utils import (
    ParamInfo,
    CandidateType,
    steps_to_pretty_str,
    time_to_pretty_str,
    parse_timeout,
)
import time


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


class ProgBar(Callback):
    def __init__(self, schedule, run_history, disable):
        self._schedule = schedule
        self._run_history = run_history
        self.disabled = disable
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
        self._last_refreshed = time.time()

    def on_search_start(self, search):
        if not self.disabled:
            self._tqdm.write(f"Search is scheduled for {self._schedule.to_total_str()}")

    def on_evaluate_end(self, new_best: dict, f: float, info: ParamInfo):
        self.update()

    def on_evaluate_canceled(self, candidate: dict, info: ParamInfo):
        self.update()

    def update(self, close=False):
        self._tqdm.n = (
            self._schedule.total_units if close else self._schedule.current_units
        )
        self._tqdm.set_postfix_str(self._str_time_per_eval(), refresh=False)
        if self._run_history.best_f is not None:
            self._tqdm.set_description_str(
                f"Best f: {self._run_history.best_f:0.3g} (out of {self._run_history.total_amount} params)",
                refresh=False,
            )
        # TODO: Maybe there is some more elegant way implemented in tqdm
        if close or time.time() - self._last_refreshed > 0.2:
            self._last_refreshed = time.time()
            self._tqdm.refresh()

    # Endless mode:
    # Endless (stop with CTRL+C)      best: 0.42 (out of 3213) [2.3min/param]
    # Step
    # 48% xXXXXXXXxxxxxxxxxxxxxxxxxxxxx | best: 0.42 (out of 3213) (00:38<1:00) [2.3min/param]
    # Time mode
    # 48% xXXXXXXXxxxxxxxxxxxxxxxxxxxxx | best: 0.42 (out of 3213) (00:38<1:00) [2.3min/param]
    def _str_time_per_eval(self):
        total_params_evaluated = (
            self._run_history.total_amount + self._run_history.total_canceled
        )
        if total_params_evaluated == 0:
            return "..."
        seconds_per_param = self._schedule.current_runtime / total_params_evaluated
        if seconds_per_param > 60 * 60:
            return f"{60*60/seconds_per_param:0.1f} param/h"
        elif seconds_per_param > 60:
            return f"{60/seconds_per_param:0.1f} param/min"
        else:
            return f"{1/seconds_per_param:0.1f} param/s"

    def on_search_end(self):
        self.update(True)
        self._tqdm.close()
        self._pretty_print_results()

    def _pretty_print_results(self):
        text_value_quadtuple = [
            (
                "Initial solution ",
                self._run_history.best_per_type[CandidateType.INIT],
                self._run_history.amount_per_type[CandidateType.INIT],
                self._run_history.canceled_per_type[CandidateType.INIT],
                self._run_history.runtime_per_type[CandidateType.INIT],
            )
        ]
        if self._run_history.amount_per_type[CandidateType.MANUALLY_ADDED] > 0:
            text_value_quadtuple.append(
                (
                    "Manually added ",
                    self._run_history.best_per_type[CandidateType.MANUALLY_ADDED],
                    self._run_history.amount_per_type[CandidateType.MANUALLY_ADDED],
                    self._run_history.canceled_per_type[CandidateType.MANUALLY_ADDED],
                    self._run_history.runtime_per_type[CandidateType.MANUALLY_ADDED],
                )
            )
        if self._run_history.amount_per_type[CandidateType.RANDOM_SEEDING] > 0:
            text_value_quadtuple.append(
                (
                    "Random seeding",
                    self._run_history.best_per_type[CandidateType.RANDOM_SEEDING],
                    self._run_history.amount_per_type[CandidateType.RANDOM_SEEDING],
                    self._run_history.canceled_per_type[CandidateType.RANDOM_SEEDING],
                    self._run_history.runtime_per_type[CandidateType.RANDOM_SEEDING],
                )
            )
        if self._run_history.amount_per_type[CandidateType.LOCAL_SAMPLING] > 0:
            text_value_quadtuple.append(
                (
                    "Local sampling",
                    self._run_history.best_per_type[CandidateType.LOCAL_SAMPLING],
                    self._run_history.amount_per_type[CandidateType.LOCAL_SAMPLING],
                    self._run_history.canceled_per_type[CandidateType.LOCAL_SAMPLING],
                    self._run_history.runtime_per_type[CandidateType.LOCAL_SAMPLING],
                )
            )
        text_value_quadtuple.append(
            (
                "Total",
                self._run_history.best_f,
                self._run_history.total_amount,
                self._run_history.total_canceled,
                self._schedule.current_runtime,
            )
        )
        text_list = []
        for text, f, steps, canceled, elapsed in text_value_quadtuple:
            value = "x" if f is None else f"{f:0.3g}"
            text_list.append(
                [
                    text,
                    value,
                    steps_to_pretty_str(steps),
                    steps_to_pretty_str(canceled),
                    time_to_pretty_str(elapsed),
                ]
            )
        text_list.insert(0, ["Mode", "Best f", "Steps", "Canceled", "Time"])
        text_list.insert(1, ["----------------", "----", "----", "----", "----"])
        text_list.insert(-1, ["----------------", "----", "----", "----", "----"])
        if self._run_history.total_canceled == 0:
            # No candidate was canceled so let's not show this column
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


class RunHistory(Callback):
    """
    Keeps track of internal statistics for each call of ```run```, i.e., what is printed at the end of run
    """

    def __init__(self, direction):
        self._direction = direction
        self.total_runtime = 0
        self.total_amount = 0
        self.total_canceled = 0
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
        self.canceled_per_type = {
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

    def on_evaluate_canceled(self, candidate: dict, info: ParamInfo):
        self.canceled_per_type[info.type] += 1
        self.total_canceled += 1

    def on_search_start(self, search):
        self.best_f = search.best_f
        self.best_per_type[CandidateType.INIT] = self.best_f

    def on_evaluate_end(self, candidate: dict, f: float, info: ParamInfo):
        runtime = info.finished_at - info.sampled_at
        if self.is_better(self.best_f, f):
            self.best_f = f
        self.total_amount += 1
        self.total_runtime += runtime
        if self.is_better(self.best_per_type[info.type], f):
            self.best_per_type[info.type] = f
        self.amount_per_type[info.type] += 1
        self.runtime_per_type[info.type] += runtime

        ema = 0.5
        self.estimated_candidate_runtime = (
            ema * self.estimated_candidate_runtime + (1 - ema) * runtime
        )


class RunContext:
    def __init__(
        self,
        direction,
        canceler,
        ignore_nans,
        schedule,
        callbacks,
        task_executor,
        quiet,
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
        self.direction = direction
        self.canceler = canceler
        if self.canceler is not None:
            self.canceler.direction = self.direction
        self.run_history = RunHistory(self.direction)
        self.ignore_nans = ignore_nans
        self.schedule = schedule
        self.callbacks = self.hook_callbacks(callbacks)
        if not quiet:
            pbar = ProgBar(schedule, self.run_history, disable=quiet)
            self.callbacks.append(pbar)
        # Must be the first callback
        self.callbacks.insert(0, self.run_history)
        self.task_executor = task_executor

    @staticmethod
    def hook_callbacks(callbacks):
        if callbacks is None:
            return []
        if not isinstance(callbacks, list):
            # Convert single callback object to a list of size 1
            callbacks = [callbacks]
        return callbacks