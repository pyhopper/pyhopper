import pickle
import os

import pyhopper
from pyhopper.utils import ParamInfo, convert_to_checkpoint_path
import time


class Callback:
    def on_search_start(self, search: "pyhopper.Search"):
        """Called at the beginning of the search

        :param search: `pyhopper.Search` object handling the search
        """
        pass

    def on_evaluate_start(self, candidate: dict, info: ParamInfo):
        """Called after `candidate` was sampled and scheduled for evaluation

        :param candidate: Parameter value of the candidate to be evaluated
        """
        pass

    def on_evaluate_end(self, candidate: dict, f: float, info: ParamInfo):
        """Called after `candidate` was successfully evaluated

        :param candidate: Parameter value of the evaluated candidate
        :param f: Value of the objective function corresponding to the candidate
        """
        pass

    def on_duplicate_sampled(self, candidate: dict, info: ParamInfo):
        """Called if `candidate` was sampled twice

        :param candidate: Parameter value of the sampled candidate
        """
        pass

    def on_evaluate_pruned(self, candidate: dict, info: ParamInfo):
        """Called if `candidate` was pruned (by an :meth:`pyhopper.pruners.Pruner`)

        :param candidate: Parameter value of the pruned candidate
        """
        pass

    def on_evaluate_nan(self, candidate: dict, info: ParamInfo):
        """Called if `candidate` is evaluated to NaN and the `ignore_nans` argument of `run` was set to True

        :param candidate: Parameter value that evaluate to NaN
        """
        pass

    def on_new_best(self, new_best: dict, f: float, info: ParamInfo):
        """Called when a new best parameter is found

        :param new_best: Value of the new best parameter
        :param f: Value of the objective function corresponding to the new best parameter
        """
        pass

    def on_search_end(self):
        """Called at the end of the search process"""
        pass

    def state_dict(self):
        """Called when a checkpoint of the hyperparameter search is created to backup the state
        :return A dict containing the internal (runtime) state of the callback. None if the callback has no state"""
        return None

    def load_state_dict(self, state_dict):
        """
        Restores the internal state of the callback.
        :param state_dict: A dict created by the self.state_dict method
        """
        pass


class CheckpointCallback(Callback):
    def __init__(self, checkpoint_path):
        self._checkpoint_path = convert_to_checkpoint_path(checkpoint_path)
        self._search_obj = None

    @property
    def checkpoint_path(self):
        return self._checkpoint_path

    def on_search_start(self, search: "pyhopper.Search"):
        self._search_obj = search
        if os.path.isfile(self._checkpoint_path):
            self._search_obj.load(self._checkpoint_path)
            self._search_obj._run_context.pbar.write(
                f"Restored search from checkpoint '{self._checkpoint_path}'"
            )

    def on_evaluate_end(self, candidate: dict, f: float, info: ParamInfo):
        self._search_obj.save(self._checkpoint_path)

    def on_evaluate_pruned(self, candidate: dict, info: ParamInfo):
        self._search_obj.save(self._checkpoint_path)

    def on_evaluate_nan(self, candidate: dict, info: ParamInfo):
        self._search_obj.save(self._checkpoint_path)

    def on_new_best(self, new_best: dict, f: float, info: ParamInfo):
        self._search_obj.save(self._checkpoint_path)

    def on_search_end(self):
        self._search_obj.save(
            self._checkpoint_path
        )  # Search is done -> forget run context


class History(Callback):
    """
    Public API for the history of the search. Can be used by the user for plotting and analyzing the search space.
    Persistent over several consecutive calls of ```run```
    """

    def __init__(self, log_candidates=True):
        self._log_candidate_enabled = log_candidates
        self._log_candidate = []
        self._log_types = []
        self._log_f = []
        self._log_finished_at = []
        self._log_best_f = []
        self._log_runtime = []

        self._pruned_types = []
        self._pruned_candidates = []
        self._pruned_finished_at = []
        self._pruned_runtime = []
        self._nan_types = []
        self._nan_candidates = []
        self._nan_finished_at = []
        self._nan_runtime = []
        self._start_time = time.time()
        self._current_best_f = None
        self._enabled = True

    def state_dict(self):
        return {
            "log_candidate": self._log_candidate,
            "log_types": self._log_types,
            "log_f": self._log_f,
            "log_finished_at": self._log_finished_at,
            "log_best_f": self._log_best_f,
            "log_runtime": self._log_runtime,
            "pruned_types": self._pruned_types,
            "pruned_candidates": self._pruned_candidates,
            "pruned_finished_at": self._pruned_finished_at,
            "pruned_runtime": self._pruned_runtime,
            "nan_types": self._nan_types,
            "nan_candidates": self._nan_candidates,
            "nan_finished_at": self._nan_finished_at,
            "nan_runtime": self._nan_runtime,
            "start_time": self._start_time,
            "current_best_f": self._current_best_f,
        }

    def load_state_dict(self, state_dict):
        self._log_candidate = state_dict["log_candidate"]
        self._log_types = state_dict["log_types"]
        self._log_f = state_dict["log_f"]
        self._log_finished_at = state_dict["log_finished_at"]
        self._log_runtime = state_dict["log_runtime"]
        self._pruned_types = state_dict["pruned_types"]
        self._pruned_candidates = state_dict["pruned_candidates"]
        self._pruned_finished_at = state_dict["pruned_finished_at"]
        self._pruned_runtime = state_dict["pruned_runtime"]
        self._nan_types = state_dict["nan_types"]
        self._nan_candidates = state_dict["nan_candidates"]
        self._nan_finished_at = state_dict["nan_finished_at"]
        self._nan_runtime = state_dict["nan_runtime"]
        self._start_time = state_dict["start_time"]
        self._current_best_f = state_dict["current_best_f"]

    def on_search_start(self, search: "pyhopper.Search"):
        self._current_best_f = search.best_f

    def on_evaluate_pruned(self, candidate: dict, info: ParamInfo):
        runtime = info.finished_at - info.sampled_at
        self._pruned_runtime.append(runtime)
        self._pruned_types.append(info.type)
        self._pruned_finished_at.append(info.finished_at - self._start_time)

        if self._log_candidate_enabled:
            self._pruned_candidates.append(candidate)

    def on_evaluate_nan(self, candidate: dict, info: ParamInfo):
        runtime = info.finished_at - info.sampled_at
        self._nan_runtime.append(runtime)
        self._nan_types.append(info.type)
        self._nan_finished_at.append(info.finished_at - self._start_time)

        if self._log_candidate_enabled:
            self._nan_candidates.append(candidate)

    def on_evaluate_end(self, candidate: dict, f: float, info: ParamInfo):
        runtime = info.finished_at - info.sampled_at

        self._log_types.append(info.type)
        self._log_f.append(f)
        self._log_finished_at.append(info.finished_at - self._start_time)
        self._log_best_f.append(self._current_best_f)
        self._log_runtime.append(runtime)

        if self._log_candidate_enabled:
            self._log_candidate.append(candidate)

    def on_new_best(self, new_best: dict, f: float, info: ParamInfo):
        self._current_best_f = f
        self._log_best_f[-1] = f  # Overwrite retrospectively

    def get_marginal(self, item):
        if not self._log_candidate_enabled:
            raise ValueError(
                "Did not store candidates as log_candidates=False was passed to __init__"
            )
        if len(self._log_candidate) > 0:
            if item not in self._log_candidate[0].keys():
                raise ValueError(
                    f"Error: Could not find key '{item}' in logged parameters"
                )
        return [self._log_candidate[i][item] for i in range(len(self._log_candidate))]

    def get_pruned_marginal(self, item):
        if len(self._pruned_candidates) > 0:
            if item not in self._pruned_candidates[0].keys():
                raise ValueError(
                    f"Error: Could not find key '{item}' in logged parameters"
                )
        return [
            self._pruned_candidates[i][item]
            for i in range(len(self._pruned_candidates))
        ]

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._log_candidate[item]
        else:
            return self.get_marginal(item)

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
        return self._log_finished_at

    @property
    def minutes(self):
        return [t / 60 for t in self._log_finished_at]

    @property
    def hours(self):
        return [t / 60 / 60 for t in self._log_finished_at]

    def __repr__(self):
        repr_str = f"pyhopper.callbacks.History(len={len(self)}"
        if len(self) > 0:
            repr_str += f", best={self.best_f:0.3g}"
        repr_str += ")"
        return repr_str

    def clear(self):
        self._log_candidate = []
        self._log_types = []
        self._log_f = []
        self._log_finished_at = []
        self._log_best_f = []
        self._log_runtime = []

        self._pruned_types = []
        self._pruned_candidates = []
        self._pruned_finished_at = []
        self._pruned_runtime = []
        self._start_time = time.time()


class SaveBestOnDisk(Callback):
    def __init__(self, filename=None, dir=None):
        if dir is not None and filename is not None:
            raise ValueError("Cannot specify filename and dir at the same time.")
        if dir is not None:
            os.makedirs(dir, exist_ok=True)
            for i in range(10000):
                filename = os.path.join(dir, f"run_{i:05d}.pkl")
                if not os.path.isfile(filename):
                    break

        self._filename = filename

    @property
    def filename(self):
        return self._filename

    def on_new_best(self, new_best, f):
        with open(self._filename, "wb") as f:
            pickle.dump(new_best, f)
