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
import signal
import time

import loky
import os
from types import FunctionType, GeneratorType
import numpy as np
import sys

_signals_received = 0


class SignalListener:
    def __init__(self):
        self._sigterm_received = 0
        self._force_quit_callback = None
        self._gradual_quit_callback = None
        self._original_sigint_handler = None

    def signal_handler(self, sig, frame):
        global _signals_received
        _signals_received += 1
        if self._sigterm_received == 0:
            print(
                "CTRL+C received. Will terminate once the currently running candidates finished"
            )
        elif self._sigterm_received == 1:
            print(f"Will force termination on 2/3 signals")
        else:
            print(f"Terminate")
        self._sigterm_received += 1
        if self._sigterm_received >= 3:
            if self._force_quit_callback is not None:
                self._force_quit_callback()

            sys.exit(-1)
        else:
            if self._gradual_quit_callback is not None:
                self._gradual_quit_callback()

    def register_signal(self, gradual_quit_callback, force_quit_callback):
        self._gradual_quit_callback = gradual_quit_callback
        self._force_quit_callback = force_quit_callback
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.signal_handler)

    def unregister_signal(self):
        signal.signal(signal.SIGINT, self._original_sigint_handler)
        self._force_quit_callback = None
        self._gradual_quit_callback = None
        self._original_sigint_handler = None
        self._sigterm_received = 0


class CancelEvaluation(Exception):
    pass


def get_gpu_list():
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        gpus_list = [s for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        return gpus_list
    import subprocess

    res = subprocess.run("nvidia-smi -L", shell=True, stdout=subprocess.PIPE)
    gpus = res.stdout.decode("utf-8").split("\n")
    gpu_list = []
    for gpu in gpus:
        if gpu.startswith("GPU"):
            gpu_list.append(len(gpu_list))
    return gpu_list


class GPUAllocator:
    def __init__(self):
        gpu_list = get_gpu_list()
        self.num_gpus = len(gpu_list)
        self._free_gpus = gpu_list
        self._allocated = {}

    def free(self, key):
        gpu = self._allocated[key]
        self._free_gpus.append(gpu)
        del self._allocated[key]

    def pop_free(self):
        gpu = self._free_gpus.pop()
        return gpu

    def alloc(self, key, gpu):
        self._allocated[key] = gpu


class EvaluationResult:
    def __init__(self):
        self.value = None
        self.was_cancelled = None
        self.cancelled_by_user = False
        self.cancelled_by_nan = False
        self.intermediate_results = None


def dummy_signal_handler(sig, frame):
    global _signals_received
    _signals_received += 1

    if _signals_received >= 3:
        time.sleep(1.0)
        os.kill()


def execute(objective_function, candidate, canceller, kwargs, remote=False, gpu=None):

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if remote:
        signal.signal(signal.SIGINT, dummy_signal_handler)

    eval_result = EvaluationResult()
    try:
        iter_or_result = objective_function(candidate, **kwargs)
        if iter_or_result is None:
            raise ValueError(
                "Objective function returned 'None'. This probably means you forgot to add a 'return' (or 'yield') statement at the end of the function."
            )
        if isinstance(iter_or_result, GeneratorType):
            repeat = True
            eval_result.intermediate_results = []
            while repeat:
                try:
                    ir = next(iter_or_result)
                    eval_result.intermediate_results.append(ir)
                    eval_result.value = ir
                    if np.isnan(ir):
                        eval_result.was_cancelled = True
                        eval_result.cancelled_by_nan = True
                        repeat = False
                    if canceller is not None and canceller.should_cancel(
                        eval_result.intermediate_results
                    ):
                        # Let's not continue from here on
                        eval_result.was_cancelled = True
                        repeat = False
                except StopIteration:
                    repeat = False
        else:
            eval_result.value = iter_or_result
            if np.isnan(iter_or_result):
                eval_result.was_cancelled = True
                eval_result.cancelled_by_nan = True
    except CancelEvaluation:
        # If objective function raises this error, the evaluation will be treated as being cancelled
        eval_result.was_cancelled = True
        # we may need the information if the cancellation was done by the user inside the objective function
        # or by an EarlyCanceller
        eval_result.cancelled_by_user = True
    return eval_result


class TaskManager:
    def __init__(self, n_jobs, mp_backend):
        admissible_mp_backends = [
            "auto",
            "loky",
            "multiprocessing",
            "dask",
            "dask-cuda",
        ]
        if mp_backend not in admissible_mp_backends:
            raise ValueError(
                f"Unknown multiprocessing backend '{mp_backend}'. Valid options are {str(admissible_mp_backends)}"
            )
        self._gpu_allocator = None
        self._pending_candidates = []
        self._pending_futures = []
        if n_jobs == "per_gpu":
            if mp_backend in ["auto", "dask-cuda"]:
                mp_backend = "dask-cuda"
            elif mp_backend == "multiprocessing":
                # workaround if dask-cuda does not work
                self._gpu_allocator = GPUAllocator()
                n_jobs = self._gpu_allocator.num_gpus
                self._queue_max = n_jobs
            else:
                raise ValueError(
                    f"Cannot use mp_backend ```{mp_backend}``` when using per_gpu mode. Valid options are 'dask-cuda' and 'multiprocessing' ('auto' defaults to 'dask-cuda')"
                )
        elif isinstance(n_jobs, int):
            if mp_backend == "auto":
                if sys.version_info.major == 3 and sys.version_info.minor <= 6:
                    # on python 3.6 default is loky
                    mp_backend = "loky"
                else:
                    # on python 3.7 or newer default is built-in multiprocessing
                    mp_backend = "multiprocessing"
            if n_jobs <= 0:
                n_jobs = len(os.sched_getaffinity(0))
            self._queue_max = n_jobs
        else:
            raise ValueError(
                "Could not parse ```n_jobs``` argument. Valid options are positive integers, -1 (all CPU cores), and 'per_gpu'"
            )

        if mp_backend == "dask-cuda":
            try:
                from dask_cuda import LocalCUDACluster
                from dask.distributed import Client, wait
            except:
                raise ValueError(
                    "Could not import cuda-dask. Make sure dask is installed ```pip3 install -U cuda-dask```. "
                    + str(sys.exc_info()[0])
                )
            self._backend_FIRST_COMPLETED = "FIRST_COMPLETED"
            self._backend_ALL_COMPLETED = "ALL_COMPLETED"
            cluster = LocalCUDACluster()
            print("CUDA_VISIBLE_DEVICES:", cluster.cuda_visible_devices)
            self._queue_max = len(cluster.cuda_visible_devices)
            self._backend_task_executor = Client(cluster)
            self._backend_wait_func = wait
        elif mp_backend == "loky":
            self._backend_wait_func = loky.wait
            self._backend_FIRST_COMPLETED = loky.FIRST_COMPLETED
            self._backend_ALL_COMPLETED = loky.ALL_COMPLETED
            self._backend_task_executor = loky.get_reusable_executor(
                max_workers=n_jobs, timeout=20
            )
        elif mp_backend == "multiprocessing":
            import concurrent.futures

            self._backend_wait_func = concurrent.futures.wait
            self._backend_FIRST_COMPLETED = concurrent.futures.FIRST_COMPLETED
            self._backend_ALL_COMPLETED = concurrent.futures.ALL_COMPLETED
            self._backend_task_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs
            )
        elif mp_backend == "dask":
            try:
                from dask.distributed import Client, LocalCluster, wait
            except:
                raise ValueError(
                    "Could not import dask. Make sure dask is installed ```pip3 install -U dask[distributed]```."
                    + str(sys.exc_info()[0])
                )
            self._backend_FIRST_COMPLETED = "FIRST_COMPLETED"
            self._backend_ALL_COMPLETED = "ALL_COMPLETED"
            cluster = LocalCluster(n_workers=n_jobs)
            self._backend_task_executor = Client(cluster)
            self._backend_wait_func = wait
        else:
            assert False, "This should never happen"
        self._mp_backend = mp_backend

    def shutdown(self):
        self._backend_task_executor.shutdown(wait=False)

    def submit(self, objective_function, candidate_type, candidate, canceller, kwargs):
        gpu_arg = None
        if self._gpu_allocator is not None:
            # Get a free GPU
            gpu_arg = self._gpu_allocator.pop_free()
        res = self._backend_task_executor.submit(
            execute,
            objective_function,
            candidate,
            canceller,
            kwargs,
            True,
            gpu_arg,
        )
        if self._gpu_allocator is not None:
            # Mark GPU as allocated by the future object
            self._gpu_allocator.alloc(res, gpu_arg)
        self._pending_futures.append(res)
        self._pending_candidates.append((candidate_type, time.time(), candidate))

    def wait_for_first_to_complete(self):
        if len(self._pending_futures) <= 0:
            # Nothing to do
            return
        self._backend_wait_func(
            self._pending_futures, return_when=self._backend_FIRST_COMPLETED
        )

    def wait_for_all_to_complete(self):
        if len(self._pending_futures) <= 0:
            # Nothing to do
            return
        self._backend_wait_func(
            self._pending_futures, return_when=self._backend_ALL_COMPLETED
        )

    def iterate_done_tasks(self):
        i = 0
        while i < len(self._pending_futures):
            if self._pending_futures[i].done():
                if self._gpu_allocator is not None:
                    # print(
                    #     "free GPU: ",
                    #     self._gpu_allocator._allocated[self._pending_futures[i]],
                    # )
                    self._gpu_allocator.free(self._pending_futures[i])
                candidate_type, start_time, candidate = self._pending_candidates[i]
                future = self._pending_futures[i]
                self._pending_futures.pop(i)
                self._pending_candidates.pop(i)
                yield candidate_type, candidate, time.time() - start_time, future.result()
            else:
                i += 1

    @property
    def is_full(self):
        return len(self._pending_futures) >= self._queue_max