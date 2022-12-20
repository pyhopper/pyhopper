from typing import Optional, Sequence

import wandb
from datetime import datetime
import os
import pickle

from pyhopper.callbacks import Callback


class WandbCallback(Callback):
    def __init__(
        self,
        project: Optional[str] = None,
        entity: Optional[str] = None,
        group: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[Sequence] = None,
    ):
        wandb.init(
            project=project,
            entity=entity,
            name=name,
            group=group,
            tags=tags,
            job_type="pyhopper",
        )

        self._best_params = None

    def on_search_start(self, search):
        pass
        # wandb.config.update(search.current_run_config)

    def on_evaluate_start(self, candidate, param_info):
        pass

    def on_evaluate_end(self, candidate, f, param_info):
        wandb.log({"sampled_f": f})

    def on_new_best(self, new_best, f, param_info):
        self._best_params = new_best

        wandb.log({"best_f": f})

    def on_search_end(self):
        os.makedirs("pyhopper_runs", exist_ok=True)
        filename = os.path.join(
            "pyhopper_runs", datetime.now().strftime("best_params_%Y%m%d_%H%M%S.pkl")
        )
        with open(filename, "wb") as f:
            pickle.dump(self._best_params, f)
        artifact = wandb.Artifact("best_params", type="parameters")
        artifact.add_file(filename)
        wandb.log_artifact(artifact)  # Creates `animals:v0`
        wandb.finish()