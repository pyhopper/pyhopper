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

import os
from traceback import format_exception
from types import GeneratorType
import numpy as np
import sys
import subprocess
from xml.etree.ElementTree import fromstring
from .pruners.pruners import (
    set_global_pruner,
    should_prune,
    get_intermediate_results_list,
)
from .utils import unwrap_sample

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


class PruneEvaluation(Exception):
    pass


def parse_nvidia_smi():
    """
    Parse the IDs, names and compute mode of GPUs on the current machine
    :return: A triple of a list of all device IDs, a dict mapping device ID to GPU name,
    and a dict mapping device ID to a bool specifying if the device is configured in the Default compute mode.
    returns (None,None,None) if the output of nvidia-smi cannot be parsed
    """
    res = subprocess.run("nvidia-smi -L", shell=True, stdout=subprocess.PIPE)
    res = res.stdout.decode("utf-8")
    if not res.startswith("GPU"):
        return None, None, None
    lines = res.split("\n")

    gpu_ids = []
    gpu_names = {}
    gpu_default_modes = {}
    for line in lines:
        if len(line.strip()) == 0:
            continue
        idx = line.find(":")
        if idx <= 4:
            raise ValueError("Could not parse output of 'nvidia-smi -L'")
        gpu_id = line[4:idx]
        gpu_ids.append(gpu_id)
        idx2 = line.find("(UUID")
        if idx2 <= idx + 3:
            raise ValueError("Could not parse output of 'nvidia-smi -L'")
        gpu_name = line[idx + 2 : idx2 - 1]
        gpu_names[gpu_id] = gpu_name
        gpu_default_modes[gpu_id] = True
        # print(f"GPU [{gpu_id}]: {gpu_name} in mode {compute_mode}")
    return gpu_ids, gpu_names, gpu_default_modes


# def parse_nvidia_smi():
#     """
#     Parse the IDs, names and compute mode of GPUs on the current machine
#     :return: A triple of a list of all device IDs, a dict mapping device ID to GPU name,
#     and a dict mapping device ID to a bool specifying if the device is configured in the Default compute mode.
#     returns (None,None,None) if the output of nvidia-smi cannot be parsed
#     """
#     res = subprocess.run("nvidia-smi -q -x", shell=True, stdout=subprocess.PIPE)
#     res = res.stdout.decode("utf-8")
#     if not res.startswith("<?xml"):
#         return None, None, None
#     root = fromstring(res)
#     gpu_ids = []
#     gpu_names = {}
#     gpu_default_modes = {}
#     for gpu_node in root.findall("gpu"):
#         gpu_id = gpu_node.find("minor_number").text
#         gpu_ids.append(gpu_id)
#         compute_mode = gpu_node.find("compute_mode").text
#         gpu_name = gpu_node.find("product_name").text
#         gpu_names[gpu_id] = gpu_name
#         gpu_default_modes[gpu_id] = compute_mode == "Default"
#         # print(f"GPU [{gpu_id}]: {gpu_name} in mode {compute_mode}")
#     return gpu_ids, gpu_names, gpu_default_modes


def get_gpu_list():
    gpu_ids, gpu_names, gpu_default_modes = parse_nvidia_smi()
    if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
        # User specified which GPUs to use (could be a subset of the GPUs on the machine)
        if gpu_ids is None:
            print(
                "Warning: Could not parse output of 'nvidia-smi'. This probably means you need to upgrade the nvidia "
                "device drivers or reboot the machine. "
            )
        gpus_used = [s for s in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
    else:
        if gpu_ids is None:
            raise ValueError(
                "Error: Could not parse output of 'nvidia-smi'. Either specify the GPU ids via the "
                "'CUDA_VISIBLE_DEVICES' environment variable, or upgrade the nvidia device drivers and reboot the "
                "machine. "
            )
        # No explicit GPU defined -> let's use them all
        gpus_used = gpu_ids
    supports_multi_instance = True
    if gpu_ids is not None:
        for gpu in gpus_used:
            if not gpu_default_modes[gpu]:
                supports_multi_instance = False
    return gpus_used, supports_multi_instance


class GPUAllocator:
    def __init__(self, factor):
        physical_gpus, supports_multi_instance = get_gpu_list()
        if factor > 1 and not supports_multi_instance:
            raise ValueError(
                "Error: GPUs are configured in Process Exclusive mode. Cannot run multiple training processes per GPU."
            )
        virtual_gpus = []
        for i in range(factor):
            virtual_gpus.extend(physical_gpus)
        self.num_gpus = len(virtual_gpus)
        self._free_gpus = virtual_gpus
        self._allocated = {}  # dict future -> str

    def free(self, future):
        gpu = self._allocated[future]
        self._free_gpus.append(gpu)
        del self._allocated[future]

    def pop_free(self):
        gpu = self._free_gpus.pop()
        return gpu

    def alloc(self, future, gpu):
        self._allocated[future] = gpu


class EvaluationResult:
    def __init__(self):
        self.value = None
        self.was_pruned = None
        self.is_nan = False
        self.intermediate_results = None
        self.error = None


def dummy_signal_handler(sig, frame):
    global _signals_received
    _signals_received += 1

    # if _signals_received >= 3:
    #     time.sleep(1.0)
    # os.kill()


def execute(objective_function, candidate, pruner, kwargs, remote=False, gpu=None):
    """
    Wrapper function for the objective function that takes care of GPU allocation and the SIGINT signal handle
    """

    if gpu is not None:
        # GPU Mode -> Get environment variable to be consumed by PyTorch and TensorFlow
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if remote:
        signal.signal(signal.SIGINT, dummy_signal_handler)

    # Remove index auxiliary variables before calling the objective function
    candidate = candidate.unwrapped_value

    set_global_pruner(pruner)
    eval_result = EvaluationResult()
    try:
        iter_or_result = objective_function(candidate, **kwargs)
        if iter_or_result is None:
            raise ValueError(
                "Objective function returned 'None'. This probably means you forgot to add a 'return' (or 'yield') "
                "statement at the end of the function. If you intended to prune the evaluation, you can do that by "
                "\n```\nraise pyhopper.PruneEvaluation()\n```\n"
            )
        if isinstance(iter_or_result, GeneratorType):
            # Generator mode
            repeat = True
            while repeat:
                try:
                    ir = next(iter_or_result)
                    eval_result.value = ir
                    if np.isnan(ir):
                        eval_result.is_nan = True
                        repeat = False
                    if should_prune(ir):
                        # Let's not continue from here on
                        eval_result.was_pruned = True
                        repeat = False
                except StopIteration:
                    repeat = False
        else:
            # No iterator but a simple function
            eval_result.value = iter_or_result
            if np.isnan(iter_or_result):
                eval_result.is_nan = True
    except PruneEvaluation:
        # If objective function raises this error, the evaluation will be treated as being pruned
        eval_result.was_pruned = True
    except:
        etype, value, tb = sys.exc_info()
        eval_result.error = "".join(format_exception(etype, value, tb, 4096))
    eval_result.intermediate_results = get_intermediate_results_list()
    return eval_result


def parse_factor(n_jobs):
    # Remove all non-digits
    only_digits = "".join(c for c in n_jobs if c.isdigit())
    if len(only_digits) == 0:
        return 1
    return int(only_digits)


class TaskManager:
    def __init__(self, n_jobs, mp_backend):
        admissible_mp_backends = [
            "auto",
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
        if isinstance(n_jobs, str) and (
            n_jobs.endswith("per_gpu") or n_jobs.endswith("per-gpu")
        ):
            if mp_backend == "dask-cuda":
                mp_backend = "dask-cuda"
            elif mp_backend in ["auto", "multiprocessing"]:
                mp_backend = "multiprocessing"
                # workaround if dask-cuda does not work
                factor = parse_factor(n_jobs)
                self._gpu_allocator = GPUAllocator(factor)
                n_jobs = self._gpu_allocator.num_gpus
                self._queue_max = n_jobs
            else:
                raise ValueError(
                    f"Cannot use mp_backend ```{mp_backend}``` when using per_gpu mode. "
                    "Valid options are 'dask-cuda' and 'multiprocessing' ('auto' defaults to 'multiprocessing')"
                )
        else:
            n_jobs = int(n_jobs)
            if mp_backend == "auto":
                mp_backend = "multiprocessing"
            if n_jobs <= 0:
                n_jobs = len(os.sched_getaffinity(0))
            self._queue_max = n_jobs

        if mp_backend == "dask-cuda":
            # Experimental
            try:
                from dask_cuda import LocalCUDACluster
                from dask.distributed import Client, wait
            except ImportError:
                raise ValueError(
                    "Could not import cuda-dask. Make sure dask is installed ```pip3 install -U cuda-dask```. "
                    + str(sys.exc_info()[0])
                )
            self._backend_FIRST_COMPLETED = "FIRST_COMPLETED"
            self._backend_ALL_COMPLETED = "ALL_COMPLETED"
            cluster = LocalCUDACluster()
            # print("CUDA_VISIBLE_DEVICES:", cluster.cuda_visible_devices)
            self._queue_max = len(cluster.cuda_visible_devices)
            self._backend_task_executor = Client(cluster)
            self._backend_wait_func = wait
            n_jobs = -1
        elif mp_backend == "multiprocessing":
            import concurrent.futures

            self._backend_wait_func = concurrent.futures.wait
            self._backend_FIRST_COMPLETED = concurrent.futures.FIRST_COMPLETED
            self._backend_ALL_COMPLETED = concurrent.futures.ALL_COMPLETED
            self._backend_task_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=n_jobs
            )
        elif mp_backend == "dask":
            # Experimental
            try:
                from dask.distributed import Client, LocalCluster, wait
            except ImportError:
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
        self.n_jobs = n_jobs

    def shutdown(self):
        self._backend_task_executor.shutdown(wait=False)

    def submit(self, objective_function, candidate, param_info, pruner, kwargs):
        gpu_arg = None
        if self._gpu_allocator is not None:
            # GPU Mode -> Get a free GPU
            gpu_arg = self._gpu_allocator.pop_free()
        res = self._backend_task_executor.submit(
            execute,
            objective_function,
            candidate,
            pruner,
            kwargs,
            True,  # remote = True
            gpu_arg,
        )
        if self._gpu_allocator is not None:
            # GPU Mode -> Mark GPU as allocated by the future object
            self._gpu_allocator.alloc(res, gpu_arg)
        self._pending_futures.append(res)
        self._pending_candidates.append((candidate, param_info))

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
        """
        Generator that gathers all finished tasks.
        yields tuples of the form (type,param,runtime, result_f)
        """
        i = 0
        while i < len(self._pending_futures):
            if self._pending_futures[i].done():
                if self._gpu_allocator is not None:
                    # GPU Mode -> Mark GPU as freed
                    self._gpu_allocator.free(self._pending_futures[i])
                candidate, param_info = self._pending_candidates[i]
                param_info.finished_at = time.time()
                future = self._pending_futures[i]
                self._pending_futures.pop(i)
                self._pending_candidates.pop(i)
                yield candidate, param_info, future.result()
            else:
                i += 1

    @property
    def is_full(self):
        return len(self._pending_futures) >= self._queue_max