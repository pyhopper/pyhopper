import pickle
import os


class Callback:
    def on_search_start(self, search):
        pass

    def on_evaluate_start(self, candidate):
        pass

    def on_evaluate_end(self, candidate, f):
        pass

    def on_evaluate_cancelled(self, candidate):
        pass

    def on_new_best(self, new_best, f):
        pass

    def on_search_end(self, history):
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