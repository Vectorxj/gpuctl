from __future__ import annotations

import argparse
from collections import deque
import json
import os
import queue
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Sequence, TextIO

from . import __version__
from .api_client import APIClient, APIError, TransportError
from .common import DEFAULT_SOCKET_PATH, cards_argument, duration_argument
from .skill_installer import SkillInstallError, install_skill


INFRASTRUCTURE_EXIT_CODE = 125


class ClientFailure(Exception):
    pass


class SignalAbort(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum


class SignalState:
    def __init__(self) -> None:
        self.event = threading.Event()
        self._pending: deque[int] = deque()
        self._old_handlers: dict[int, signal.Handlers] = {}

    def __enter__(self) -> "SignalState":
        def handle(signum: int, _frame: object) -> None:
            self._pending.append(signum)
            self.event.set()

        watched = [signal.SIGINT, signal.SIGTERM]
        for name in ("SIGHUP", "SIGQUIT"):
            if hasattr(signal, name):
                watched.append(getattr(signal, name))
        for signum in watched:
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, handle)
        return self

    def __exit__(self, *_args: object) -> None:
        for signum, handler in self._old_handlers.items():
            signal.signal(signum, handler)

    def pop(self) -> int | None:
        if not self._pending:
            self.event.clear()
            return None
        signum = self._pending.popleft()
        if not self._pending:
            self.event.clear()
        return signum


class LeaseHeartbeat:
    def __init__(
        self,
        api: APIClient,
        lease: dict[str, Any],
        *,
        request_timeout: float,
    ) -> None:
        self._api = api
        self._lease_id = str(lease["lease_id"])
        self._ttl = float(lease["ttl_seconds"])
        remaining = float(lease.get("remaining_seconds", self._ttl))
        self._safe_remaining = self._with_safety_margin(remaining)
        self._request_timeout = request_timeout
        self._stop = threading.Event()
        self._failures: queue.Queue[Exception] = queue.Queue(maxsize=1)
        self._thread = threading.Thread(
            target=self._run,
            name="gpuctl-lease-heartbeat",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self._request_timeout + 1.0)

    def failure(self) -> Exception | None:
        try:
            return self._failures.get_nowait()
        except queue.Empty:
            return None

    def _run(self) -> None:
        safe_deadline = time.monotonic() + self._safe_remaining
        next_attempt = time.monotonic() + min(
            1.0,
            self._ttl / 3.0,
            self._safe_remaining / 2.0,
        )
        last_error: Exception | None = None

        while not self._stop.is_set():
            now = time.monotonic()
            if now >= safe_deadline:
                self._fail(last_error or ClientFailure("lease renewal deadline passed"))
                return
            wake_at = min(next_attempt, safe_deadline)
            if self._stop.wait(max(0.0, wake_at - now)):
                return
            if time.monotonic() >= safe_deadline:
                self._fail(last_error or ClientFailure("lease renewal deadline passed"))
                return

            started = time.monotonic()
            timeout = min(self._request_timeout, max(0.05, safe_deadline - started))
            try:
                lease = self._api.renew_lease(self._lease_id, timeout=timeout)
                self._ttl = float(lease["ttl_seconds"])
                remaining = float(lease.get("remaining_seconds", self._ttl))
                safe_deadline = started + self._with_safety_margin(remaining)
                next_attempt = time.monotonic() + min(1.0, self._ttl / 3.0)
                last_error = None
            except APIError as error:
                last_error = error
                if error.status < 500:
                    self._fail(error)
                    return
                next_attempt = time.monotonic() + min(1.0, self._ttl / 10.0)
            except TransportError as error:
                last_error = error
                next_attempt = time.monotonic() + min(1.0, self._ttl / 10.0)
            except (KeyError, TypeError, ValueError) as error:
                self._fail(ClientFailure(f"invalid lease renewal response: {error}"))
                return

    def _with_safety_margin(self, remaining: float) -> float:
        margin = max(0.5, min(2.0, self._ttl * 0.2))
        return max(0.1, remaining - margin)

    def _fail(self, error: Exception) -> None:
        try:
            self._failures.put_nowait(error)
        except queue.Full:
            pass


def _default_owner() -> str:
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    return f"{user}@{socket.gethostname()}:{os.getpid()}"


def _add_connection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--socket",
        default=os.environ.get("GPUCTL_SOCKET", DEFAULT_SOCKET_PATH),
        help=f"gpuctld Unix socket (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--request-timeout",
        type=duration_argument,
        default=10.0,
        metavar="DURATION",
        help="timeout for each HTTP operation (default: 10s)",
    )


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuctl",
        description="Acquire GPU cards, run a command, and release the cards.",
        epilog=(
            "Other commands: gpuctl status, gpuctl jobs, "
            "gpuctl cancel TASK_ID, gpuctl install-skill"
        ),
    )
    _add_connection_arguments(parser)
    allocation = parser.add_mutually_exclusive_group(required=True)
    allocation.add_argument(
        "--cards",
        type=cards_argument,
        metavar="ID,ID,...",
        help="wait for these exact card IDs",
    )
    allocation.add_argument(
        "--count",
        "--gpus",
        dest="count",
        type=int,
        metavar="N",
        help="allocate N currently free cards",
    )
    parser.add_argument("--owner", default=_default_owner(), help="owner label")
    parser.add_argument(
        "--wait-timeout",
        type=duration_argument,
        default=0.0,
        metavar="DURATION",
        help="maximum queue wait; 0 means no limit (default: 0)",
    )
    parser.add_argument(
        "--poll-interval",
        type=duration_argument,
        default=0.5,
        metavar="DURATION",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--kill-after",
        type=duration_argument,
        default=5.0,
        metavar="DURATION",
        help="force-kill a command this long after lease loss (default: 5s)",
    )
    parser.add_argument(
        "--set-cuda-visible-devices",
        action="store_true",
        help="set CUDA_VISIBLE_DEVICES to the allocated card IDs",
    )
    parser.add_argument("--quiet", action="store_true", help="suppress progress messages")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def build_status_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuctl status",
        description="Show cards, active leases, and the wait queue.",
    )
    _add_connection_arguments(parser)
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    return parser


def build_jobs_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuctl jobs",
        description="List your queued and running GPU tasks.",
    )
    _add_connection_arguments(parser)
    parser.add_argument(
        "--all",
        action="store_true",
        help="list every user's tasks (requires uid 0)",
    )
    parser.add_argument("--json", action="store_true", help="print raw JSON")
    return parser


def build_cancel_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuctl cancel",
        description="Cancel one of your tasks; uid 0 may cancel any task.",
    )
    _add_connection_arguments(parser)
    parser.add_argument("task_id", help="task ID shown by gpuctl status or jobs")
    return parser


def build_install_skill_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuctl install-skill",
        description="Install the bundled GPU lease skill for coding agents.",
    )
    parser.add_argument(
        "--agent",
        choices=("all", "copilot", "claude"),
        default="all",
        help="agent configuration to install (default: all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing modified skill",
    )
    return parser


def _api_from_args(args: argparse.Namespace) -> APIClient:
    if args.request_timeout <= 0:
        raise ClientFailure("--request-timeout must be positive")
    try:
        return APIClient(
            args.socket,
            timeout=args.request_timeout,
        )
    except ValueError as error:
        raise ClientFailure(str(error)) from error


def _acquire(
    api: APIClient,
    args: argparse.Namespace,
    signals: SignalState,
    stderr: TextIO,
    command: list[str],
) -> dict[str, Any]:
    if args.count is not None and args.count <= 0:
        raise ClientFailure("--count must be positive")
    if args.wait_timeout < 0:
        raise ClientFailure("--wait-timeout must not be negative")
    if args.poll_interval <= 0:
        raise ClientFailure("--poll-interval must be positive")

    command_display = shlex.join(command)
    if len(command_display) > 2048:
        command_display = command_display[:2045] + "..."
    state = api.create_request(
        owner=args.owner,
        cards=args.cards,
        count=args.count,
        command=command_display,
    )
    request_id = str(state["request_id"])
    acquired = False
    deadline = (
        time.monotonic() + args.wait_timeout if args.wait_timeout > 0 else None
    )
    last_contact = time.monotonic()
    queue_ttl = float(state.get("queue_ttl_seconds", 30.0))
    last_position: int | None = None

    try:
        while True:
            if state.get("status") == "granted":
                lease = state.get("lease")
                if not isinstance(lease, dict):
                    raise ClientFailure("server returned a granted request without a lease")
                acquired = True
                return lease
            if state.get("status") != "queued":
                raise ClientFailure(f"server returned unknown request state: {state!r}")

            position = int(state["position"])
            if not args.quiet and position != last_position:
                print(
                    f"[gpuctl] task {request_id} queued at position {position}",
                    file=stderr,
                )
                last_position = position

            signum = signals.pop()
            if signum is not None:
                raise SignalAbort(signum)

            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise ClientFailure("timed out waiting for GPU cards")
            wait_for = args.poll_interval
            if deadline is not None:
                wait_for = min(wait_for, max(0.0, deadline - now))
            if signals.event.wait(wait_for):
                continue

            try:
                state = api.get_request(request_id)
                last_contact = time.monotonic()
                queue_ttl = float(state.get("queue_ttl_seconds", queue_ttl))
            except TransportError:
                if time.monotonic() - last_contact >= queue_ttl:
                    raise ClientFailure(
                        "lost contact with gpuctld long enough for the queue ticket to expire"
                    )
            except APIError as error:
                if error.status == 404:
                    raise ClientFailure("queue request expired or was removed") from error
                raise
    finally:
        if not acquired:
            try:
                api.cancel_request(request_id)
            except (APIError, TransportError):
                pass


def _release(api: APIClient, lease_id: str) -> Exception | None:
    try:
        api.release_lease(lease_id)
        return None
    except (APIError, TransportError) as error:
        return error


def _terminate_process(process: subprocess.Popen[bytes], kill_after: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=kill_after)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _exit_code(returncode: int) -> int:
    if returncode >= 0:
        return min(returncode, 255)
    return min(128 + abs(returncode), 255)


def _execute(
    api: APIClient,
    lease: dict[str, Any],
    command: list[str],
    args: argparse.Namespace,
    signals: SignalState,
    stderr: TextIO,
) -> int:
    lease_id = str(lease["lease_id"])
    task_id = str(lease["task_id"])
    cards = [str(card) for card in lease["cards"]]
    if not args.quiet:
        print(
            f"[gpuctl] task {task_id} running on cards: {','.join(cards)}",
            file=stderr,
        )

    pending_signal = signals.pop()
    if pending_signal is not None:
        release_error = _release(api, lease_id)
        if release_error is not None:
            print(f"gpuctl: failed to release lease: {release_error}", file=stderr)
        return 128 + pending_signal

    try:
        lease = api.renew_lease(lease_id)
    except (APIError, TransportError, KeyError, TypeError, ValueError) as error:
        release_error = _release(api, lease_id)
        print(f"gpuctl: could not confirm lease before command start: {error}", file=stderr)
        if release_error is not None:
            print(f"gpuctl: failed to release lease: {release_error}", file=stderr)
        return INFRASTRUCTURE_EXIT_CODE

    environment = os.environ.copy()
    environment["GPUCTL_CARDS"] = ",".join(cards)
    environment["GPUCTL_TASK_ID"] = task_id
    environment["GPUCTL_LEASE_ID"] = lease_id
    if args.set_cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(cards)

    try:
        process = subprocess.Popen(
            command,
            env=environment,
            start_new_session=True,
        )
    except FileNotFoundError as error:
        release_error = _release(api, lease_id)
        print(f"gpuctl: command not found: {command[0]}", file=stderr)
        if release_error is not None:
            print(f"gpuctl: failed to release lease: {release_error}", file=stderr)
        return 127
    except PermissionError:
        release_error = _release(api, lease_id)
        print(f"gpuctl: command is not executable: {command[0]}", file=stderr)
        if release_error is not None:
            print(f"gpuctl: failed to release lease: {release_error}", file=stderr)
        return 126
    except OSError as error:
        release_error = _release(api, lease_id)
        print(f"gpuctl: could not start command: {error}", file=stderr)
        if release_error is not None:
            print(f"gpuctl: failed to release lease: {release_error}", file=stderr)
        return 126

    heartbeat = LeaseHeartbeat(
        api,
        lease,
        request_timeout=args.request_timeout,
    )
    heartbeat.start()
    result = 0
    lease_lost = False
    try:
        while True:
            returncode = process.poll()
            if returncode is not None:
                result = _exit_code(returncode)
                break

            renewal_error = heartbeat.failure()
            if renewal_error is not None:
                lease_lost = True
                if (
                    isinstance(renewal_error, APIError)
                    and renewal_error.code == "task_canceled"
                ):
                    print(f"gpuctl: task {task_id} was canceled", file=stderr)
                else:
                    print(f"gpuctl: lease renewal failed: {renewal_error}", file=stderr)
                _terminate_process(process, args.kill_after)
                result = INFRASTRUCTURE_EXIT_CODE
                break

            signum = signals.pop()
            if signum is not None:
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass
            try:
                process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass
    finally:
        heartbeat.stop()

    release_error = _release(api, lease_id)
    if release_error is not None:
        print(f"gpuctl: failed to release lease: {release_error}", file=stderr)
        if result == 0 and not lease_lost:
            result = INFRASTRUCTURE_EXIT_CODE
    return result


def _run(args: argparse.Namespace, stderr: TextIO) -> int:
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        raise ClientFailure("a command is required after --")
    if args.kill_after < 0:
        raise ClientFailure("--kill-after must not be negative")

    api = _api_from_args(args)
    with SignalState() as signals:
        lease = _acquire(api, args, signals, stderr, command)
        return _execute(api, lease, command, args, signals, stderr)


def _format_duration(value: Any) -> str:
    if value is None:
        return "-"
    seconds = max(0, int(float(value)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _task_request(task: dict[str, Any]) -> str:
    if task.get("status") in {"running", "canceling"}:
        return ",".join(str(card) for card in task.get("cards", []))
    if "requested_cards" in task:
        return "want:" + ",".join(
            str(card) for card in task.get("requested_cards", [])
        )
    return f"want:{task.get('requested_count', '?')}"


def _print_tasks(tasks: list[dict[str, Any]], stdout: TextIO) -> None:
    if not tasks:
        print("No active tasks.", file=stdout)
        return
    print(
        f"{'TASK ID':<16} {'USER':<12} {'STATE':<8} {'CARDS':<14} "
        f"{'QUEUED':<8} {'RUNNING':<8} COMMAND",
        file=stdout,
    )
    for task in tasks:
        command = str(task.get("command") or "-")
        if len(command) > 100:
            command = command[:97] + "..."
        print(
            f"{str(task.get('task_id', '')):<16} "
            f"{str(task.get('user', '')):<12} "
            f"{str(task.get('status', '')):<8} "
            f"{_task_request(task):<14} "
            f"{_format_duration(task.get('queued_for_seconds')):<8} "
            f"{_format_duration(task.get('running_for_seconds')):<8} "
            f"{command}",
            file=stdout,
        )


def _print_status(status: dict[str, Any], stdout: TextIO) -> None:
    print(
        f"{'CARD':<16} {'STATE':<8} {'USER':<12} {'TASK ID':<16} EXPIRES",
        file=stdout,
    )
    for card in status.get("cards", []):
        print(
            f"{str(card.get('id', '')):<16} "
            f"{str(card.get('state', '')):<8} "
            f"{str(card.get('user', '-')):<12} "
            f"{str(card.get('task_id', '-')):<16} "
            f"{str(card.get('expires_at', '-'))}",
            file=stdout,
        )
    print("\nTASKS", file=stdout)
    _print_tasks(status.get("tasks", []), stdout)


def _status(args: argparse.Namespace, stdout: TextIO) -> int:
    api = _api_from_args(args)
    status = api.status()
    if args.json:
        json.dump(status, stdout, indent=2, sort_keys=True)
        print(file=stdout)
    else:
        _print_status(status, stdout)
    return 0


def _jobs(args: argparse.Namespace, stdout: TextIO) -> int:
    api = _api_from_args(args)
    result = api.tasks(include_all=args.all)
    if args.json:
        json.dump(result, stdout, indent=2, sort_keys=True)
        print(file=stdout)
    else:
        _print_tasks(result.get("tasks", []), stdout)
    return 0


def _cancel(args: argparse.Namespace, stdout: TextIO) -> int:
    api = _api_from_args(args)
    api.cancel_task(args.task_id)
    print(f"Canceled task {args.task_id}.", file=stdout)
    return 0


def _install_skill(args: argparse.Namespace, stdout: TextIO) -> int:
    results = install_skill(args.agent, force=args.force)
    for result in results:
        action = "installed" if result.changed else "already up to date"
        print(
            f"{result.agent}: {action}: {result.destination}",
            file=stdout,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    try:
        if arguments and arguments[0] == "status":
            args = build_status_parser().parse_args(arguments[1:])
            return _status(args, sys.stdout)
        if arguments and arguments[0] == "jobs":
            args = build_jobs_parser().parse_args(arguments[1:])
            return _jobs(args, sys.stdout)
        if arguments and arguments[0] == "cancel":
            args = build_cancel_parser().parse_args(arguments[1:])
            return _cancel(args, sys.stdout)
        if arguments and arguments[0] == "install-skill":
            args = build_install_skill_parser().parse_args(arguments[1:])
            return _install_skill(args, sys.stdout)
        args = build_run_parser().parse_args(arguments)
        return _run(args, sys.stderr)
    except SignalAbort as error:
        return min(128 + error.signum, 255)
    except (
        ClientFailure,
        SkillInstallError,
        APIError,
        TransportError,
        KeyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"gpuctl: {error}", file=sys.stderr)
        return INFRASTRUCTURE_EXIT_CODE
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
