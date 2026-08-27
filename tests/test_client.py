from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from gpuctl.api_client import APIClient
from gpuctl.client import main
from gpuctl.http_api import GPUUnixHTTPServer
from gpuctl.manager import LeaseManager


class ClientIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.socket_directory = tempfile.TemporaryDirectory()
        self.socket_path = str(
            Path(self.socket_directory.name) / "gpuctld.sock"
        )
        self.manager = LeaseManager(("2", "5"), lease_ttl=3.0, queue_ttl=3.0)
        self.server = GPUUnixHTTPServer(self.socket_path, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.manager.close()
        self.socket_directory.cleanup()

    def test_runs_command_with_allocated_environment_and_releases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "environment.txt"
            script = (
                "import os, pathlib; "
                f"pathlib.Path({str(output)!r}).write_text("
                "os.environ['GPUCTL_CARDS'] + '\\n' + "
                "os.environ['GPUCTL_TASK_ID'], encoding='utf-8')"
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                result = main(
                    [
                        "--socket",
                        self.socket_path,
                        "--count",
                        "2",
                        "--",
                        sys.executable,
                        "-c",
                        script,
                    ]
                )

            self.assertEqual(result, 0, stderr.getvalue())
            cards, task_id = output.read_text(encoding="utf-8").splitlines()
            self.assertEqual(cards, "2,5")
            self.assertEqual(len(task_id), 16)
            self.assertEqual(self.manager.status()["summary"]["free_cards"], 2)

    def test_preserves_command_exit_code_and_releases(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            result = main(
                [
                    "--socket",
                    self.socket_path,
                    "--cards",
                    "5",
                    "--",
                    sys.executable,
                    "-c",
                    "raise SystemExit(7)",
                ]
            )
        self.assertEqual(result, 7)
        self.assertEqual(self.manager.status()["summary"]["free_cards"], 2)

    def test_renews_lease_during_long_command(self) -> None:
        started = time.monotonic()
        with contextlib.redirect_stderr(io.StringIO()):
            result = main(
                [
                    "--socket",
                    self.socket_path,
                    "--cards",
                    "2",
                    "--",
                    sys.executable,
                    "-c",
                    "import time; time.sleep(3.3)",
                ]
            )
        self.assertEqual(result, 0)
        self.assertGreaterEqual(time.monotonic() - started, 3.0)
        self.assertEqual(self.manager.status()["summary"]["free_cards"], 2)

    def test_leaves_existing_cuda_visible_devices_unchanged_by_default(self) -> None:
        original = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = "original"
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "visible.txt"
                script = (
                    "import os, pathlib; "
                    f"pathlib.Path({str(output)!r}).write_text("
                    "os.environ['CUDA_VISIBLE_DEVICES'], encoding='utf-8')"
                )
                with contextlib.redirect_stderr(io.StringIO()):
                    result = main(
                        [
                            "--socket",
                            self.socket_path,
                            "--cards",
                            "2",
                            "--",
                            sys.executable,
                            "-c",
                            script,
                        ]
                    )
                self.assertEqual(result, 0)
                self.assertEqual(output.read_text(encoding="utf-8"), "original")
        finally:
            if original is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original

    def test_can_set_cuda_visible_devices_explicitly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "visible.txt"
            script = (
                "import os, pathlib; "
                f"pathlib.Path({str(output)!r}).write_text("
                "os.environ['CUDA_VISIBLE_DEVICES'], encoding='utf-8')"
            )
            with contextlib.redirect_stderr(io.StringIO()):
                result = main(
                    [
                        "--socket",
                        self.socket_path,
                        "--cards",
                        "5",
                        "--set-cuda-visible-devices",
                        "--",
                        sys.executable,
                        "-c",
                        script,
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(output.read_text(encoding="utf-8"), "5")

    def test_status_shows_task_command_state_and_durations(self) -> None:
        api = APIClient(self.socket_path, timeout=1.0)
        running = api.create_request(
            owner="training",
            cards=("2",),
            count=None,
            command="python train.py",
        )
        queued = api.create_request(
            owner="evaluation",
            cards=("2",),
            count=None,
            command="python eval.py",
        )
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            result = main(["status", "--socket", self.socket_path])

        output = stdout.getvalue()
        self.assertEqual(result, 0)
        self.assertIn(running["task_id"], output)
        self.assertIn(queued["task_id"], output)
        self.assertIn("python train.py", output)
        self.assertIn("python eval.py", output)
        self.assertIn("running", output)
        self.assertIn("queued", output)
        self.assertIn("QUEUED", output)
        self.assertIn("RUNNING", output)

        api.cancel_request(queued["task_id"])
        api.release_lease(running["lease"]["lease_id"])

    def test_cancel_command_stops_running_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            started = Path(directory) / "started"
            script = (
                "import pathlib,time; "
                f"pathlib.Path({str(started)!r}).write_text('started'); "
                "time.sleep(30)"
            )
            environment = os.environ.copy()
            source = str(Path(__file__).resolve().parents[1] / "src")
            environment["PYTHONPATH"] = source
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "gpuctl",
                    "--socket",
                    self.socket_path,
                    "--cards",
                    "2",
                    "--quiet",
                    "--",
                    sys.executable,
                    "-c",
                    script,
                ],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            self.addCleanup(self._stop_process, process)

            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not started.exists():
                time.sleep(0.02)
            self.assertTrue(started.exists(), "GPU command did not start")

            api = APIClient(self.socket_path, timeout=1.0)
            tasks = api.tasks()["tasks"]
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0]["status"], "running")
            self.assertIn("time.sleep(30)", tasks[0]["command"])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                result = main(
                    [
                        "cancel",
                        "--socket",
                        self.socket_path,
                        tasks[0]["task_id"],
                    ]
                )
            self.assertEqual(result, 0)
            self.assertIn(tasks[0]["task_id"], stdout.getvalue())

            _stdout, stderr = process.communicate(timeout=5.0)
            returncode = process.returncode
            self.assertEqual(returncode, 125, stderr)
            self.assertIn("was canceled", stderr)
            self.assertEqual(self.manager.status()["summary"]["free_cards"], 2)

    @staticmethod
    def _stop_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
