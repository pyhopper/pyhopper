import pickle
import os


class Callback:
    def on_search_start(self, search):
        """Called at the beginning of the search

        :param search: `pyhopper.Search` object handling the search
        """
        pass

    def on_evaluate_start(self, candidate: dict):
        """Called after `candidate` was sampled and scheduled for evaluation

        :param candidate: Parameter value of the candidate to be evaluated
        """
        pass

    def on_evaluate_end(self, candidate, f):
        """Called after `candidate` was successfully evaluated

        :param candidate: Parameter value of the evaluated candidate
        :param f: Value of the objective function corresponding to the candidate
        """
        pass

    def on_evaluate_cancelled(self, candidate):
        """Called if `candidate` was cancelled (by an :meth:`pyhopper.cancellers.EarlyCanceller`)

        :param candidate: Parameter value of the cancelled candidate
        """
        pass

    def on_new_best(self, new_best, f):
        """Called when a new best parameter is found

        :param new_best: Value of the new best parameter
        :param f: Value of the objective function corresponding to the new best parameter
        """
        pass

    def on_search_end(self, history):
        """Called at the end of the search process

        :param history: pyhopper.History object containing the history of all evaluated parameters
        """
        pass


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