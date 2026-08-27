# gpuctl

`gpuctl` is a lightweight, cooperative GPU locking tool for multiple users on a
single machine. `gpuctld` maintains leases and a strict FIFO queue locally on the
GPU host. Users and coding agents queue through `gpuctl`, which acquires the
required lock before starting a command and releases it after the command exits.

This is **cooperative allocation**, not container or kernel-level isolation. The
daemon does not inspect GPU utilization or other processes. Users and agents are
responsible for ensuring that commands use only their allocated cards.

## Components

| Component | Purpose |
|---|---|
| `gpuctld` | Local daemon that manages tasks, card locks, the FIFO queue, and fallback lease recovery |
| `gpuctl` | Acquires cards, runs commands, lists or cancels tasks, renews leases, and releases cards |
| `gpuctl-gpu` skill | Instructs Copilot, Claude Code, and other agents to use `gpuctl` before GPU work |
| `contrib/gpuctld.service` | systemd service unit |

Runtime requires Python 3.10 or later and has no third-party Python dependencies.

## Build

Run the tests and build a wheel from the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m pip wheel . --no-deps -w dist
```

The resulting artifact is:

```text
dist/gpuctl-0.1.0-py3-none-any.whl
```

The wheel contains `gpuctl`, `gpuctld`, and the agent skill. Publish it as a
GitHub Release asset, upload it to an internal artifact registry, or copy it
directly to the GPU host.

## Install or upgrade on the GPU host

Run the installer from the repository root as a regular user:

```bash
./install.sh
```

The script:

1. Runs the test suite and builds a wheel in `dist/`.
2. Installs or reinstalls that wheel in `/opt/gpuctl`.
3. Links `gpuctl` and `gpuctld` into `/usr/local/bin`.
4. Creates the daemon configuration on the first install, installs the systemd
   unit, and restarts `gpuctld`.
5. Refreshes the packaged Copilot and Claude skills for the user running the
   installer.

If that user already has a pip user installation at `~/.local/bin/gpuctl`, the
script refreshes it from the same wheel so it cannot shadow the machine-wide
command with stale code.

The first install detects numeric `/dev/nvidiaN` devices automatically. Pass an
explicit ordered card list to override detection or change an existing
configuration:

```bash
./install.sh --cards 0,1,2,3
```

Subsequent runs preserve `/etc/gpuctl/gpuctld.env` unless `--cards` is supplied.
The installer refuses to restart the daemon while tasks are active or queued.
Wait until it is idle, or use `--force-restart` only when intentionally
discarding all in-memory tasks and locks during coordinated maintenance.
Already-running workloads are not killed and would continue without tracked
leases.

Useful options:

```bash
./install.sh --skip-tests
./install.sh --build-only
./install.sh --skip-skill
```

Do not run the script with `sudo`; it requests administrator access only for
the machine-wide files and service. Other users can use `gpuctl` immediately
afterward. Each user who wants the agent integration must separately run:

```bash
gpuctl install-skill
```

The default skill refresh uses `--force`. Use `--skip-skill` if the invoking
user has locally modified either packaged skill.

### Manual package installation

#### Recommended: use a dedicated virtual environment

This approach does not modify system Python packages:

```bash
sudo python3 -m venv /opt/gpuctl
sudo /opt/gpuctl/bin/python -m pip install ./gpuctl-0.1.0-py3-none-any.whl
sudo ln -sfn /opt/gpuctl/bin/gpuctl /usr/local/bin/gpuctl
sudo ln -sfn /opt/gpuctl/bin/gpuctld /usr/local/bin/gpuctld
gpuctl --version
gpuctld --version
```

On Debian or Ubuntu, install the system package first if `venv` is unavailable:

```bash
sudo apt-get install python3-venv
```

If the host permits administrator-managed packages in the system Python
installation, the wheel can instead be installed directly:

```bash
sudo python3 -m pip install ./gpuctl-0.1.0-py3-none-any.whl
```

This is a machine-wide installation. Individual users do not need separate
program installations because `/usr/local/bin/gpuctl` is available to everyone.

## Manual daemon deployment

The daemon must be configured with the cards it manages. Their order is also the
selection order used for automatic `--count` allocations.

Create the configuration:

```bash
sudo install -d -m 0755 /etc/gpuctl
printf 'GPUCTL_CARDS=0,1,2,3\n' | sudo tee /etc/gpuctl/gpuctld.env
```

Install and start the systemd service:

```bash
sudo install -m 0644 contrib/gpuctld.service /etc/systemd/system/gpuctld.service
sudo systemctl daemon-reload
sudo systemctl enable --now gpuctld
sudo systemctl status gpuctld
```

By default, `gpuctld` creates `/run/gpuctl/gpuctld.sock` with mode `0666`, so
every local user can connect. The daemon obtains the caller's real UID and PID
from Linux `SO_PEERCRED` instead of trusting a client-provided username:

- Regular users can cancel only tasks created by the same UID.
- UID 0 can list and cancel tasks owned by any user.
- No shared token is needed because communication does not use TCP.

The systemd service uses `DynamicUser`. It does not need access to GPU devices
because it manages only cooperative locks in memory.

To run the daemon in the foreground without systemd:

```bash
gpuctld --cards 0,1,2,3 --socket /tmp/gpuctld.sock
export GPUCTL_SOCKET=/tmp/gpuctld.sock
```

Restarting the service removes all in-memory tasks and locks. The default lease
TTL and queue-ticket TTL are both 30 seconds. Change them with `--lease-ttl` and
`--queue-ttl`.

## Install the skill for Copilot and Claude Code

The wheel bundles the `gpuctl-gpu` skill. Personal skills live in each user's
home directory, so every user on the machine should run this once:

```bash
gpuctl install-skill
```

By default, the command installs both:

```text
~/.copilot/skills/gpuctl-gpu/SKILL.md
~/.claude/skills/gpuctl-gpu/SKILL.md
```

Install for only one agent with:

```bash
gpuctl install-skill --agent copilot
gpuctl install-skill --agent claude
```

The installer will not overwrite a modified skill. Explicitly replace an older
version during an upgrade:

```bash
gpuctl install-skill --force
```

### Verify in GitHub Copilot CLI

In a new session, run:

```text
/skills info gpuctl-gpu
```

If the skill was installed during an existing session, reload skills first:

```text
/skills reload
```

The skill can also be listed from the shell:

```bash
copilot skill list
```

### Verify in Claude Code

Restart Claude Code after installation, then enter:

```text
/gpuctl-gpu
```

The skill description causes agents to load it automatically for CUDA, ROCm,
PyTorch, TensorFlow, JAX, model training or inference, GPU benchmarks, and
GPU-backed tests. It can also be requested explicitly:

```text
Use the /gpuctl-gpu skill and run this training job.
```

This repository also contains a project skill at
`.claude/skills/gpuctl-gpu/SKILL.md`. Copilot and Claude Code can discover it
while working in this repository. `gpuctl install-skill` makes the same rule
available across all repositories for the current user.

The skill follows the
[Agent Skills specification](https://agentskills.io/specification). See the
[GitHub Copilot skills documentation](https://docs.github.com/en/copilot/how-tos/copilot-cli/customize-copilot/add-skills)
and [Claude Code skills documentation](https://code.claude.com/docs/en/skills)
for agent-specific directory behavior.

## Usage

A systemd deployment uses `/run/gpuctl/gpuctld.sock` by default and requires no
client configuration.

Request any one currently unlocked card:

```bash
gpuctl --count 1 -- python train.py
```

Atomically request any two cards:

```bash
gpuctl --count 2 -- torchrun --nproc-per-node=2 train.py
```

Wait for specific physical cards:

```bash
gpuctl --cards 1,3 -- python train.py
```

`gpuctl` sets these variables for the child process:

- `GPUCTL_CARDS`: allocated physical card IDs, such as `1,3`
- `GPUCTL_TASK_ID`: task ID used by `status` and `cancel`
- `GPUCTL_LEASE_ID`: internal lease ID

`CUDA_VISIBLE_DEVICES` is not changed by default. Set it to the allocated cards
for a CUDA application with:

```bash
gpuctl --count 2 --set-cuda-visible-devices -- \
  torchrun --nproc-per-node=2 train.py
```

The allocated physical cards are then renumbered to logical devices `0..N-1`
inside the child process.

If an application requires physical IDs, read `GPUCTL_CARDS` inside the wrapped
shell:

```bash
gpuctl --count 1 -- \
  bash -lc 'python run.py --physical-gpu "$GPUCTL_CARDS"'
```

Show card and task state:

```bash
gpuctl status
gpuctl status --json
```

`status` lists every active task:

```text
TASK ID          USER         STATE    CARDS          QUEUED   RUNNING  COMMAND
5f81aeb1639f66af alice        running  0,1            4s       12m03s   torchrun --nproc-per-node=2 train.py
a1d0778642ea94dc bob          queued   want:2         31s      -        python eval.py
```

Each task has a unique 16-character ID and one of these states:

- `queued`: waiting in the strict FIFO queue
- `running`: cards have been allocated and the command is running
- `canceling`: cancellation was requested; cards remain locked while the client
  stops the command and releases its lease

`QUEUED` is the time spent waiting before allocation. `RUNNING` is the time
elapsed since allocation. While waiting, `CARDS` shows the request; while
running, it shows the actual allocation.

List and cancel tasks owned by the current user:

```bash
gpuctl jobs
gpuctl cancel 5f81aeb1639f66af
```

As root, list or cancel tasks owned by any user:

```bash
sudo gpuctl jobs --all
sudo gpuctl cancel 5f81aeb1639f66af
```

Canceling a queued task removes it immediately. Canceling a running task first
moves it to `canceling`. The corresponding `gpuctl` wrapper then terminates the
entire child process group and releases the cards. Those cards are not assigned
to the next task before release completes.

Limit queue wait time with:

```bash
gpuctl --count 2 --wait-timeout 20m -- python train.py
```

## Agent workflow for GPU tasks

The bundled skill requires agents to follow this workflow:

1. Do not hold a lock for CPU-only installation, builds, or data preparation.
2. Use `--cards` when the user requests physical card IDs. Otherwise use
   `--count`; default to one card when no count is specified.
3. Run the entire GPU workload as a foreground child of `gpuctl`.
4. Put consecutive GPU commands in one wrapped shell so they share one lease.
5. Do not infer availability from `nvidia-smi`, and never bypass `gpuctl`
   because a request is queued or the daemon reports an error.
6. Do not detach GPU work inside the wrapper with `&`, `nohup`, or similar
   mechanisms. The lease is released when the wrapper exits.
7. To stop a task, find its ID with `gpuctl status` or `gpuctl jobs`, then run
   `gpuctl cancel TASK_ID`.

Example with consecutive GPU phases:

```bash
python prepare_dataset.py
gpuctl --count 2 --set-cuda-visible-devices -- \
  bash -lc 'python gpu_preprocess.py && torchrun --nproc-per-node=2 train.py'
```

## Scheduling and failure semantics

1. A request receives all requested cards or none. Multi-card allocations are
   atomic.
2. The queue is strict FIFO. Later requests cannot bypass a blocked head request.
3. Normal cancellation uses the owner-or-root `cancel` endpoint and does not
   depend on TTL expiry.
4. A waiting client refreshes its queue ticket. If it crashes or is killed with
   `SIGKILL`, the daemon removes the stale request after the queue TTL.
5. A healthy running client continuously renews its lease, so tasks can run
   indefinitely. The lease TTL is not a maximum runtime. It reclaims a stale
   lock only when a wrapper crashes before releasing it.
6. A canceled running task continues holding its cards until the wrapper confirms
   that the command has stopped and releases the lease. If the wrapper is
   unreachable, the lease TTL is the final fallback.
7. TTL expiry reclaims only the logical lock. The daemon does not run as root and
   cannot forcibly kill arbitrary operating-system processes that have detached
   from their wrapper. An administrator must handle that exceptional case with
   system process tools.
8. The wrapped command's exit status is preserved. Exit code `125` indicates a
   `gpuctl` infrastructure failure. Exit codes `126` and `127` indicate that the
   command is not executable or was not found.

## Upgrade

Pull or otherwise update the checkout, then rerun the installer:

```bash
git pull
./install.sh
```

The wheel is force-reinstalled even when the package version has not changed,
so rerunning the script also deploys development snapshots.
