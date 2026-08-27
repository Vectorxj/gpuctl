---
name: gpuctl-gpu
description: Ensures GPU workloads on a shared machine acquire a gpuctl lease before execution. Use before running CUDA, ROCm, PyTorch, TensorFlow, JAX, model training or inference, GPU benchmarks, GPU-backed tests, or any command that may initialize or consume a GPU.
compatibility: Requires gpuctl on PATH and a reachable gpuctld daemon on the shared GPU machine. Compatible with GitHub Copilot and Claude Code.
metadata:
  version: "0.1.0"
---

# Run GPU workloads through gpuctl

## Mandatory rule

Every command that may initialize or consume a GPU must run as a foreground child
of `gpuctl`. Acquire the lease before the first GPU operation and keep the wrapper
alive until every GPU child process exits.

Do not:

- run a GPU command directly, even as a quick test;
- infer availability from `nvidia-smi` and bypass the daemon;
- fall back to an unwrapped command when `gpuctl` is missing, the daemon is
  unavailable, or the request is queued;
- detach the wrapped workload with `&`, `nohup`, or a background launcher that
  lets the `gpuctl` child exit while GPU work continues.

If the lease cannot be obtained, stop and report the error. Read-only inspection
such as `gpuctl status`, `nvidia-smi` queries, source inspection, dependency
installation, and CPU-only builds do not need a lease unless they initialize a
GPU runtime.

## Choose the allocation

1. If the user names physical card IDs, request exactly those cards:

   ```bash
   gpuctl --cards 1,3 --set-cuda-visible-devices -- <command> [args...]
   ```

2. Otherwise request the required number of cards:

   ```bash
   gpuctl --count N --set-cuda-visible-devices -- <command> [args...]
   ```

3. If neither card IDs nor a count is specified and the workload does not clearly
   require multiple GPUs, request one card with `--count 1`.

Use multiple GPUs only when the task or workload configuration requires them.
Never submit several independent one-card requests as a substitute for one atomic
multi-card request.

## Make the workload use the lease

For CUDA-aware commands, prefer `--set-cuda-visible-devices`. The allocated
physical cards then appear to the child as logical devices `0..N-1`.

```bash
gpuctl --count 2 --set-cuda-visible-devices -- \
  torchrun --nproc-per-node=2 train.py
```

If the command must receive physical IDs or does not use
`CUDA_VISIBLE_DEVICES`, omit that option and read `GPUCTL_CARDS` inside the
wrapped command:

```bash
gpuctl --count 1 -- \
  bash -lc 'python run.py --physical-gpu "$GPUCTL_CARDS"'
```

`GPUCTL_CARDS` is a comma-separated list of allocated physical card IDs.
`GPUCTL_TASK_ID` is the ID shown by status and cancellation commands.
`GPUCTL_LEASE_ID` identifies the internal active lease.

## Keep one lease across related GPU commands

CPU-only preparation should happen before acquiring the lease. If several GPU
commands must use the same allocation continuously, put them in one foreground
shell owned by `gpuctl`:

```bash
python prepare_dataset.py
gpuctl --count 2 --set-cuda-visible-devices -- \
  bash -lc 'python gpu_preprocess.py && torchrun --nproc-per-node=2 train.py'
```

Do not start a second `gpuctl` request from inside an active lease.

## Raw timed reservations

Use a raw reservation only when the user explicitly asks to hold cards for a
fixed period without wrapping a command:

```bash
gpuctl grab --cards 1 --for 2h
gpuctl grab --count 2 --for 30m
```

The daemon retains the reservation after `grab` returns and releases it at the
fixed deadline. Do not use `grab` as a substitute for wrapping an agent-run GPU
workload: commands started outside the wrapper are not terminated when the
reservation is canceled or expires.

## Queueing and failures

- Submit the wrapped command directly; `gpuctld` is the source of truth and will
  queue it when the requested cards are unavailable.
- `gpuctl` prints a unique task ID. Use `gpuctl status` to see every active task,
  its command, queue duration, run duration, state, and cards. Use `gpuctl jobs`
  to list tasks owned by the current Unix UID.
- To stop one of the current user's tasks, run `gpuctl cancel TASK_ID`. Only use
  `sudo gpuctl cancel TASK_ID` when the user explicitly asks for administrative
  cancellation of another user's task.
- Use `--wait-timeout DURATION` only when the task has a bounded wait.
- `gpuctl` returns the workload's exit code. Exit code `125` means allocation,
  renewal, or release infrastructure failed.
- On exit code `125`, report the failure and do not rerun the GPU command without
  `gpuctl`.
- `gpuctl` renews the lease while the command runs and releases it when the
  command exits. If the client is killed abruptly, the daemon reclaims the lease
  after its TTL.
