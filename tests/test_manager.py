from __future__ import annotations

import time
import unittest

from gpuctl.manager import (
    LeaseManager,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)


class LeaseManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = LeaseManager(
            ("0", "1", "2"),
            lease_ttl=1.0,
            queue_ttl=1.0,
        )

    def tearDown(self) -> None:
        self.manager.close()

    def test_atomic_multi_card_allocation_is_strict_fifo(self) -> None:
        first = self.manager.create_request(owner="first", count=2)
        self.assertEqual(first["status"], "granted")
        self.assertEqual(first["lease"]["cards"], ["0", "1"])

        second = self.manager.create_request(owner="second", cards=["1", "2"])
        third = self.manager.create_request(owner="third", cards=["2"])
        self.assertEqual(second["status"], "queued")
        self.assertEqual(third["status"], "queued")
        self.assertEqual(third["position"], 2)

        self.manager.release_lease(first["lease"]["lease_id"])
        second = self.manager.get_request(second["request_id"])
        third = self.manager.get_request(third["request_id"])
        self.assertEqual(second["status"], "granted")
        self.assertEqual(second["lease"]["cards"], ["1", "2"])
        self.assertEqual(third["status"], "queued")

        self.manager.release_lease(second["lease"]["lease_id"])
        third = self.manager.get_request(third["request_id"])
        self.assertEqual(third["status"], "granted")
        self.assertEqual(third["lease"]["cards"], ["2"])

    def test_canceling_blocked_head_unblocks_next_request(self) -> None:
        first = self.manager.create_request(owner="holder", cards=["0"])
        blocked = self.manager.create_request(owner="blocked", cards=["0", "1"])
        next_request = self.manager.create_request(owner="next", cards=["1"])

        self.assertEqual(next_request["status"], "queued")
        self.assertTrue(self.manager.cancel_request(blocked["request_id"]))
        next_request = self.manager.get_request(next_request["request_id"])
        self.assertEqual(next_request["status"], "granted")
        self.manager.release_lease(first["lease"]["lease_id"])

    def test_expired_lease_is_reaped_and_grants_waiter(self) -> None:
        manager = LeaseManager(("0",), lease_ttl=0.08, queue_ttl=1.0)
        self.addCleanup(manager.close)
        first = manager.create_request(owner="first", cards=["0"])
        second = manager.create_request(owner="second", cards=["0"])

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            state = manager.get_request(second["request_id"])
            if state["status"] == "granted":
                break
            time.sleep(0.01)
        else:
            self.fail("queued request was not granted after lease expiry")

        with self.assertRaises(NotFoundError):
            manager.get_request(first["request_id"])

    def test_unpolled_queue_ticket_expires(self) -> None:
        manager = LeaseManager(("0",), lease_ttl=1.0, queue_ttl=0.06)
        self.addCleanup(manager.close)
        manager.create_request(owner="holder", cards=["0"])
        waiting = manager.create_request(owner="waiting", cards=["0"])
        time.sleep(0.12)
        with self.assertRaises(NotFoundError):
            manager.get_request(waiting["request_id"])

    def test_rejects_invalid_requests(self) -> None:
        with self.assertRaisesRegex(ValidationError, "exactly one"):
            self.manager.create_request(owner="x")
        with self.assertRaisesRegex(ValidationError, "unknown"):
            self.manager.create_request(owner="x", cards=["9"])
        with self.assertRaisesRegex(ValidationError, "cannot exceed"):
            self.manager.create_request(owner="x", count=4)
        with self.assertRaisesRegex(ValidationError, "duplicates"):
            self.manager.create_request(owner="x", cards=["0", "0"])

    def test_task_metadata_and_durations_are_reported(self) -> None:
        running = self.manager.create_request(
            owner="training",
            cards=["0"],
            uid=1000,
            user="alice",
            client_pid=123,
            command="python train.py",
        )
        queued = self.manager.create_request(
            owner="evaluation",
            cards=["0"],
            uid=1001,
            user="bob",
            client_pid=456,
            command="python eval.py",
        )
        time.sleep(0.02)

        tasks = self.manager.status()["tasks"]
        self.assertEqual([task["status"] for task in tasks], ["running", "queued"])
        self.assertEqual(tasks[0]["task_id"], running["task_id"])
        self.assertEqual(tasks[0]["command"], "python train.py")
        self.assertGreater(tasks[0]["running_for_seconds"], 0)
        self.assertEqual(tasks[1]["task_id"], queued["task_id"])
        self.assertEqual(tasks[1]["command"], "python eval.py")
        self.assertGreater(tasks[1]["queued_for_seconds"], 0)

    def test_only_owner_or_admin_can_cancel_task(self) -> None:
        task = self.manager.create_request(
            owner="training",
            cards=["0"],
            uid=1000,
            user="alice",
            command="python train.py",
        )
        with self.assertRaises(PermissionDeniedError):
            self.manager.request_task_cancel(task["task_id"], uid=1001)
        self.assertTrue(
            self.manager.request_task_cancel(
                task["task_id"],
                uid=1001,
                admin=True,
            )
        )
        status = self.manager.status()
        self.assertEqual(status["tasks"][0]["status"], "canceling")
        self.assertEqual(status["summary"]["active_leases"], 1)
        self.assertEqual(status["summary"]["free_cards"], 2)
        self.assertTrue(
            self.manager.cancel_request(task["task_id"], uid=1000)
        )
        self.assertEqual(self.manager.status()["summary"]["active_leases"], 0)

    def test_canceling_task_holds_cards_until_wrapper_releases(self) -> None:
        running = self.manager.create_request(
            owner="training",
            cards=["0"],
            uid=1000,
            user="alice",
            command="python train.py",
        )
        waiting = self.manager.create_request(
            owner="next",
            cards=["0"],
            uid=1001,
            user="bob",
            command="python next.py",
        )

        self.manager.request_task_cancel(running["task_id"], uid=1000)
        waiting_state = self.manager.get_request(
            waiting["task_id"],
            uid=1001,
        )
        self.assertEqual(waiting_state["status"], "queued")

        self.manager.release_lease(
            running["lease"]["lease_id"],
            uid=1000,
        )
        waiting_state = self.manager.get_request(
            waiting["task_id"],
            uid=1001,
        )
        self.assertEqual(waiting_state["status"], "granted")

    def test_timed_reservation_uses_fixed_expiry(self) -> None:
        manager = LeaseManager(("0",), lease_ttl=0.05, queue_ttl=1.0)
        self.addCleanup(manager.close)
        reservation = manager.create_request(
            owner="manual reservation",
            cards=["0"],
            uid=1000,
            user="alice",
            command="gpuctl grab --cards 0 --for 0.15s",
            reservation_seconds=0.15,
        )
        self.assertEqual(manager.status()["tasks"][0]["status"], "reserved")
        self.assertIn("reservation_ends_at", reservation["lease"])

        time.sleep(0.07)
        manager.renew_lease(reservation["lease"]["lease_id"], uid=1000)
        self.assertEqual(manager.status()["summary"]["free_cards"], 0)

        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            if manager.status()["summary"]["free_cards"] == 1:
                break
            time.sleep(0.01)
        else:
            self.fail("timed reservation did not expire")
        self.assertEqual(manager.status()["tasks"], [])

    def test_canceling_reservation_immediately_unblocks_queue(self) -> None:
        reservation = self.manager.create_request(
            owner="manual reservation",
            cards=["0"],
            uid=1000,
            user="alice",
            command="gpuctl grab --cards 0 --for 1h",
            reservation_seconds=3600,
        )
        waiting = self.manager.create_request(
            owner="training",
            cards=["0"],
            uid=1001,
            user="bob",
            command="python train.py",
        )

        self.manager.request_task_cancel(reservation["task_id"], uid=1000)
        waiting_state = self.manager.get_request(waiting["task_id"], uid=1001)
        self.assertEqual(waiting_state["status"], "granted")
