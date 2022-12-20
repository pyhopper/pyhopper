from typing import Optional, Sequence

from tensorboardX import SummaryWriter
from datetime import datetime
import os
import pickle

from pyhopper.callbacks import Callback


class TensorboardCallback(Callback):
    def __init__(
        self, logdir: Optional[str] = None, comment: str = "", flush_secs: int = 5
    ):
        self._logdir = logdir
        self._tb_writer = SummaryWriter(logdir, comment=comment, flush_secs=flush_secs)

        self._step = 0

    def on_search_start(self, search):
        pass

    def on_evaluate_start(self, candidate, param_info):
        pass

    def on_evaluate_end(self, candidate, f, param_info):
        self._tb_writer.add_scalar("sampled_f", f, self._step)
        self._step += 1

    def on_new_best(self, new_best, f, param_info):
        self._tb_writer.add_scalar("best_f", f, self._step)

    def on_search_end(self, history):
        self._tb_writer.close()