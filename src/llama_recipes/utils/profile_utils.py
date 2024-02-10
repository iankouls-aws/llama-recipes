# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import contextlib
import os
import torch


@contextlib.contextmanager
def maybe_run_profiler(config, *pos_args, **kwargs):

    # get user defined profiler settings
    if config.enable_profiler:
        dump_dir = "profile_traces"
        save_trace_dir = config.model_name
        trace_dir = os.path.join(dump_dir, save_trace_dir)
        iter_frequency = 5
        _global_iter_count = 0


        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            nonlocal _global_iter_count
            _global_iter_count += iter_frequency
            curr_trace_dir_name = "iteration_" + str(_global_iter_count)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir)
            if rank==0:
                print(f"exporting profile traces to {curr_trace_dir}")

            prof.export_chrome_trace(f"{curr_trace_dir}/rank{rank}_trace.json")

        if rank==0:
            print(f"Profiling active.  Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=iter_frequency - 2,
                warmup=1,
                active=1,
                repeat=2,
            ),
            on_trace_ready=trace_handler,
            profile_memory=True,
            with_stack=False,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None
