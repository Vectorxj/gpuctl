from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
import threading
import unittest

from gpuctl.api_client import APIClient
from gpuctl.http_api import GPUUnixHTTPServer
from gpuctl.manager import LeaseManager


class HTTPAPITest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.socket_path = str(Path(self.directory.name) / "gpuctld.sock")
        self.manager = LeaseManager(("0", "1"), lease_ttl=2.0, queue_ttl=2.0)
        self.server = GPUUnixHTTPServer(self.socket_path, self.manager)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = APIClient(self.socket_path, timeout=1.0)

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.manager.close()
        self.directory.cleanup()

    def test_full_lease_lifecycle_and_peer_identity(self) -> None:
        request = self.client.create_request(
            owner="training",
            cards=None,
            count=2,
            command="python train.py",
        )
        self.assertEqual(request["status"], "granted")
        self.assertEqual(request["uid"], os.getuid())
        self.assertEqual(len(request["task_id"]), 16)
        lease = request["lease"]
        self.assertEqual(lease["cards"], ["0", "1"])

        status = self.client.status()
        self.assertEqual(status["summary"]["active_leases"], 1)
        self.assertEqual(status["summary"]["free_cards"], 0)
        self.assertEqual(status["tasks"][0]["task_id"], request["task_id"])
        self.assertEqual(status["tasks"][0]["status"], "running")
        self.assertEqual(status["tasks"][0]["command"], "python train.py")
        self.assertIn("queued_for_seconds", status["tasks"][0])
        self.assertIn("running_for_seconds", status["tasks"][0])

        renewed = self.client.renew_lease(lease["lease_id"])
        self.assertEqual(renewed["lease_id"], lease["lease_id"])
        self.client.release_lease(lease["lease_id"])

        status = self.client.status()
        self.assertEqual(status["summary"]["active_leases"], 0)
        self.assertEqual(status["summary"]["free_cards"], 2)
        self.assertEqual(status["tasks"], [])

    def test_user_can_list_and_cancel_own_task(self) -> None:
        request = self.client.create_request(
            owner="alice",
            cards=("0",),
            count=None,
            command="python train.py",
        )
        tasks = self.client.tasks()
        self.assertEqual([task["task_id"] for task in tasks["tasks"]], [request["task_id"]])

        self.client.cancel_task(request["task_id"])
        tasks = self.client.tasks()["tasks"]
        self.assertEqual(tasks[0]["status"], "canceling")
        self.assertEqual(self.client.status()["summary"]["free_cards"], 1)

        self.client.release_lease(request["lease"]["lease_id"])
        self.assertEqual(self.client.tasks()["tasks"], [])
        self.assertEqual(self.client.status()["summary"]["free_cards"], 2)

    def test_socket_is_available_to_all_local_users(self) -> None:
        mode = stat.S_IMODE(os.stat(self.socket_path).st_mode)
        self.assertEqual(mode, 0o666)
