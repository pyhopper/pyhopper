from typing import Optional
import aim
import os

from pyhopper.callbacks import Callback


class AimCallback(Callback):
    def __init__(
        self,
        name: Optional[str] = "PyHopper",
        repo: Optional[str] = None,
    ):
        self.run: aim.Run = aim.Run(experiment=name, repo=repo)
        self._best_params = None

    def on_search_start(self, search):
        pass
        # wandb.config.update(search.current_run_config)

    def on_evaluate_start(self, candidate, param_info):
        pass

    def on_evaluate_end(self, candidate, f, param_info):
        self.run.track(f, name="sampled_f")

    def on_new_best(self, new_best, f, param_info):
        self._best_params = new_best
        self.run.track(f, name="best_f")

    def on_search_end(self):
        os.makedirs("pyhopper_runs", exist_ok=True)
        for key, value in self._best_params.items():
            self.run.set(('best_params', key), value, strict=False)
        self.run.report_successful_finish()
