from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets
import threading
import time
from typing import Any, Iterable


class ManagerError(Exception):
    """Base class for lease manager errors."""


class ValidationError(ManagerError):
    pass


class NotFoundError(ManagerError):
    pass


class ClosedError(ManagerError):
    pass


class PermissionDeniedError(ManagerError):
    pass


class TaskCanceledError(ManagerError):
    pass


def _timestamp_after(seconds: float) -> str:
    value = datetime.now(timezone.utc) + timedelta(seconds=seconds)
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


@dataclass
class _Request:
    request_id: str
    uid: int
    user: str
    client_pid: int | None
    owner: str
    command: str | None
    cards: tuple[str, ...] | None
    count: int | None
    enqueued_at: str
    enqueued_mono: float
    started_at: str | None
    started_mono: float | None
    cancel_requested_at: str | None
    canceled_by_uid: int | None
    queue_deadline: float
    queue_expires_at: str
    status: str = "queued"
    lease_id: str | None = None


@dataclass
class _Lease:
    lease_id: str
    request_id: str
    uid: int
    user: str
    owner: str
    cards: tuple[str, ...]
    deadline: float
    expires_at: str


class LeaseManager:
    """Thread-safe strict-FIFO scheduler for atomic GPU leases."""

    def __init__(
        self,
        cards: Iterable[str],
        *,
        lease_ttl: float = 30.0,
        queue_ttl: float = 30.0,
    ) -> None:
        card_order = tuple(cards)
        if not card_order or any(not isinstance(card, str) or not card for card in card_order):
            raise ValueError("at least one non-empty card ID is required")
        if len(set(card_order)) != len(card_order):
            raise ValueError("card IDs must be unique")
        if lease_ttl <= 0:
            raise ValueError("lease_ttl must be positive")
        if queue_ttl <= 0:
            raise ValueError("queue_ttl must be positive")

        self._card_order = card_order
        self._card_set = frozenset(card_order)
        self._lease_ttl = float(lease_ttl)
        self._queue_ttl = float(queue_ttl)
        self._holders: dict[str, str] = {}
        self._leases: dict[str, _Lease] = {}
        self._requests: dict[str, _Request] = {}
        self._queue: deque[str] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._reaper = threading.Thread(
            target=self._reaper_loop,
            name="gpuctl-lease-reaper",
            daemon=True,
        )
        self._reaper.start()

    @property
    def cards(self) -> tuple[str, ...]:
        return self._card_order

    def create_request(
        self,
        *,
        owner: str,
        cards: list[str] | tuple[str, ...] | None = None,
        count: int | None = None,
        uid: int = -1,
        user: str | None = None,
        client_pid: int | None = None,
        command: str | None = None,
    ) -> dict[str, Any]:
        owner_value, card_values, count_value = self._validate_request(
            owner=owner,
            cards=cards,
            count=count,
        )
        user_value, command_value = self._validate_metadata(
            uid=uid,
            user=user,
            client_pid=client_pid,
            command=command,
            owner=owner_value,
        )
        now = time.monotonic()
        with self._condition:
            if self._closed:
                raise ClosedError("lease manager is shutting down")
            self._maintain_locked(now)
            request_id = secrets.token_hex(8)
            while request_id in self._requests:
                request_id = secrets.token_hex(8)
            request = _Request(
                request_id=request_id,
                uid=uid,
                user=user_value,
                client_pid=client_pid,
                owner=owner_value,
                command=command_value,
                cards=card_values,
                count=count_value,
                enqueued_at=_timestamp_now(),
                enqueued_mono=now,
                started_at=None,
                started_mono=None,
                cancel_requested_at=None,
                canceled_by_uid=None,
                queue_deadline=now + self._queue_ttl,
                queue_expires_at=_timestamp_after(self._queue_ttl),
            )
            self._requests[request.request_id] = request
            self._queue.append(request.request_id)
            self._schedule_locked(now)
            self._condition.notify_all()
            return self._request_snapshot_locked(request, now)

    def get_request(
        self,
        request_id: str,
        *,
        uid: int | None = None,
        admin: bool = False,
    ) -> dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            self._maintain_locked(now)
            request = self._requests.get(request_id)
            if request is None:
                raise NotFoundError("request does not exist or has expired")
            self._authorize_request_locked(request, uid=uid, admin=admin)
            if request.status == "queued":
                request.queue_deadline = now + self._queue_ttl
                request.queue_expires_at = _timestamp_after(self._queue_ttl)
                self._condition.notify_all()
            return self._request_snapshot_locked(request, now)

    def cancel_request(
        self,
        request_id: str,
        *,
        uid: int | None = None,
        admin: bool = False,
    ) -> bool:
        now = time.monotonic()
        with self._condition:
            self._maintain_locked(now)
            request = self._requests.get(request_id)
            if request is None:
                return False
            self._authorize_request_locked(request, uid=uid, admin=admin)
            if request.lease_id is not None:
                self._release_lease_locked(request.lease_id)
            else:
                self._requests.pop(request_id, None)
                self._remove_from_queue_locked(request_id)
            self._schedule_locked(now)
            self._condition.notify_all()
            return True

    def renew_lease(
        self,
        lease_id: str,
        *,
        uid: int | None = None,
        admin: bool = False,
    ) -> dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            self._maintain_locked(now)
            lease = self._leases.get(lease_id)
            if lease is None:
                raise NotFoundError("lease does not exist or has expired")
            request = self._requests[lease.request_id]
            self._authorize_request_locked(request, uid=uid, admin=admin)
            if request.status == "canceling":
                raise TaskCanceledError(
                    f"task {request.request_id} has been canceled"
                )
            lease.deadline = now + self._lease_ttl
            lease.expires_at = _timestamp_after(self._lease_ttl)
            self._condition.notify_all()
            return self._lease_snapshot_locked(lease, now)

    def release_lease(
        self,
        lease_id: str,
        *,
        uid: int | None = None,
        admin: bool = False,
    ) -> bool:
        now = time.monotonic()
        with self._condition:
            self._maintain_locked(now)
            lease = self._leases.get(lease_id)
            if lease is None:
                return False
            request = self._requests[lease.request_id]
            self._authorize_request_locked(request, uid=uid, admin=admin)
            self._release_lease_locked(lease_id)
            self._schedule_locked(now)
            self._condition.notify_all()
            return True

    def request_task_cancel(
        self,
        task_id: str,
        *,
        uid: int | None = None,
        admin: bool = False,
    ) -> bool:
        now = time.monotonic()
        with self._condition:
            self._maintain_locked(now)
            request = self._requests.get(task_id)
            if request is None:
                return False
            self._authorize_request_locked(request, uid=uid, admin=admin)
            if request.status == "queued":
                self._requests.pop(task_id, None)
                self._remove_from_queue_locked(task_id)
                self._schedule_locked(now)
            elif request.status == "granted":
                request.status = "canceling"
                request.cancel_requested_at = _timestamp_now()
                request.canceled_by_uid = uid
            self._condition.notify_all()
            return True

    def status(self) -> dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            self._maintain_locked(now)
            cards = []
            for card in self._card_order:
                lease_id = self._holders.get(card)
                lease = self._leases.get(lease_id) if lease_id is not None else None
                if lease is None:
                    cards.append({"id": card, "state": "free"})
                else:
                    cards.append(
                        {
                            "id": card,
                            "state": "leased",
                            "task_id": lease.request_id,
                            "owner": lease.owner,
                            "user": lease.user,
                            "uid": lease.uid,
                            "expires_at": lease.expires_at,
                        }
                    )

            queue_items = []
            for position, request_id in enumerate(self._queue, start=1):
                request = self._requests[request_id]
                item: dict[str, Any] = {
                    "task_id": request.request_id,
                    "position": position,
                    "owner": request.owner,
                    "user": request.user,
                    "uid": request.uid,
                    "enqueued_at": request.enqueued_at,
                }
                if request.cards is not None:
                    item["cards"] = list(request.cards)
                else:
                    item["count"] = request.count
                queue_items.append(item)

            return {
                "cards": cards,
                "queue": queue_items,
                "tasks": [
                    self._task_snapshot_locked(request, now)
                    for request in self._requests.values()
                ],
                "summary": {
                    "total_cards": len(self._card_order),
                    "free_cards": sum(card["state"] == "free" for card in cards),
                    "active_leases": len(self._leases),
                    "queued_requests": len(self._queue),
                },
            }

    def tasks(
        self,
        *,
        uid: int | None = None,
        include_all: bool = False,
    ) -> dict[str, Any]:
        now = time.monotonic()
        with self._condition:
            self._maintain_locked(now)
            requests = [
                request
                for request in self._requests.values()
                if include_all or uid is None or request.uid == uid
            ]
            return {
                "tasks": [
                    self._task_snapshot_locked(request, now)
                    for request in requests
                ]
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        if threading.current_thread() is not self._reaper:
            self._reaper.join(timeout=2.0)

    def _validate_request(
        self,
        *,
        owner: str,
        cards: list[str] | tuple[str, ...] | None,
        count: int | None,
    ) -> tuple[str, tuple[str, ...] | None, int | None]:
        if not isinstance(owner, str) or not owner.strip():
            raise ValidationError("owner must be a non-empty string")
        owner_value = owner.strip()
        if len(owner_value) > 256:
            raise ValidationError("owner must be at most 256 characters")

        has_cards = cards is not None
        has_count = count is not None
        if has_cards == has_count:
            raise ValidationError("specify exactly one of cards or count")

        if has_cards:
            if not isinstance(cards, (list, tuple)) or not cards:
                raise ValidationError("cards must be a non-empty list")
            if any(not isinstance(card, str) or not card for card in cards):
                raise ValidationError("every card ID must be a non-empty string")
            card_values = tuple(cards)
            if len(set(card_values)) != len(card_values):
                raise ValidationError("cards must not contain duplicates")
            unknown = [card for card in card_values if card not in self._card_set]
            if unknown:
                raise ValidationError(f"unknown card IDs: {', '.join(unknown)}")
            return owner_value, card_values, None

        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValidationError("count must be a positive integer")
        if count > len(self._card_order):
            raise ValidationError(
                f"count cannot exceed the configured card count ({len(self._card_order)})"
            )
        return owner_value, None, count

    def _validate_metadata(
        self,
        *,
        uid: int,
        user: str | None,
        client_pid: int | None,
        command: str | None,
        owner: str,
    ) -> tuple[str, str | None]:
        if isinstance(uid, bool) or not isinstance(uid, int) or uid < -1:
            raise ValidationError("uid must be a non-negative integer")
        if user is None:
            user_value = owner
        elif not isinstance(user, str) or not user:
            raise ValidationError("user must be a non-empty string")
        else:
            user_value = user
        if len(user_value) > 256:
            raise ValidationError("user must be at most 256 characters")
        if (
            client_pid is not None
            and (
                isinstance(client_pid, bool)
                or not isinstance(client_pid, int)
                or client_pid <= 0
            )
        ):
            raise ValidationError("client_pid must be a positive integer")
        if command is not None:
            if not isinstance(command, str) or not command.strip():
                raise ValidationError("command must be a non-empty string")
            command = command.strip()
            if len(command) > 2048:
                raise ValidationError("command must be at most 2048 characters")
        return user_value, command

    def _maintain_locked(self, now: float) -> None:
        expired_leases = [
            lease_id
            for lease_id, lease in self._leases.items()
            if lease.deadline <= now
        ]
        for lease_id in expired_leases:
            self._release_lease_locked(lease_id)

        expired_requests = {
            request_id
            for request_id in self._queue
            if self._requests[request_id].queue_deadline <= now
        }
        if expired_requests:
            self._queue = deque(
                request_id
                for request_id in self._queue
                if request_id not in expired_requests
            )
            for request_id in expired_requests:
                self._requests.pop(request_id, None)

        self._schedule_locked(now)

    def _schedule_locked(self, now: float) -> None:
        while self._queue:
            request_id = self._queue[0]
            request = self._requests[request_id]
            selected_cards = self._select_cards_locked(request)
            if selected_cards is None:
                return

            self._queue.popleft()
            lease = _Lease(
                lease_id=secrets.token_urlsafe(32),
                request_id=request_id,
                uid=request.uid,
                user=request.user,
                owner=request.owner,
                cards=selected_cards,
                deadline=now + self._lease_ttl,
                expires_at=_timestamp_after(self._lease_ttl),
            )
            self._leases[lease.lease_id] = lease
            for card in selected_cards:
                self._holders[card] = lease.lease_id
            request.status = "granted"
            request.lease_id = lease.lease_id
            request.started_at = _timestamp_now()
            request.started_mono = now

    def _select_cards_locked(self, request: _Request) -> tuple[str, ...] | None:
        if request.cards is not None:
            if any(card in self._holders for card in request.cards):
                return None
            return request.cards

        free_cards = tuple(
            card for card in self._card_order if card not in self._holders
        )
        assert request.count is not None
        if len(free_cards) < request.count:
            return None
        return free_cards[: request.count]

    def _release_lease_locked(self, lease_id: str) -> None:
        lease = self._leases.pop(lease_id, None)
        if lease is None:
            return
        for card in lease.cards:
            if self._holders.get(card) == lease_id:
                self._holders.pop(card, None)
        request = self._requests.get(lease.request_id)
        if request is not None and request.lease_id == lease_id:
            self._requests.pop(lease.request_id, None)

    def _remove_from_queue_locked(self, request_id: str) -> None:
        try:
            self._queue.remove(request_id)
        except ValueError:
            pass

    def _request_snapshot_locked(
        self, request: _Request, now: float
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "request_id": request.request_id,
            "task_id": request.request_id,
            "status": request.status,
            "uid": request.uid,
            "user": request.user,
            "client_pid": request.client_pid,
            "owner": request.owner,
            "command": request.command,
            "enqueued_at": request.enqueued_at,
            "queue_ttl_seconds": self._queue_ttl,
        }
        if request.cards is not None:
            result["requested_cards"] = list(request.cards)
        else:
            result["requested_count"] = request.count

        if request.status == "queued":
            result["position"] = list(self._queue).index(request.request_id) + 1
            result["queue_expires_at"] = request.queue_expires_at
        elif request.lease_id is not None:
            lease = self._leases[request.lease_id]
            result["lease"] = self._lease_snapshot_locked(lease, now)
        return result

    def _task_snapshot_locked(
        self, request: _Request, now: float
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "task_id": request.request_id,
            "status": self._public_task_status(request),
            "uid": request.uid,
            "user": request.user,
            "client_pid": request.client_pid,
            "owner": request.owner,
            "command": request.command,
            "enqueued_at": request.enqueued_at,
            "queued_for_seconds": self._queued_duration_locked(request, now),
        }
        if request.cards is not None:
            result["requested_cards"] = list(request.cards)
        else:
            result["requested_count"] = request.count
        if request.status == "queued":
            result["position"] = list(self._queue).index(request.request_id) + 1
        elif request.lease_id is not None:
            lease = self._leases[request.lease_id]
            result["cards"] = list(lease.cards)
            result["started_at"] = request.started_at
            assert request.started_mono is not None
            result["running_for_seconds"] = max(0.0, now - request.started_mono)
            result["expires_at"] = lease.expires_at
            result["remaining_seconds"] = max(0.0, lease.deadline - now)
            if request.status == "canceling":
                result["cancel_requested_at"] = request.cancel_requested_at
                result["canceled_by_uid"] = request.canceled_by_uid
        return result

    def _queued_duration_locked(self, request: _Request, now: float) -> float:
        end = request.started_mono if request.started_mono is not None else now
        return max(0.0, end - request.enqueued_mono)

    def _public_task_status(self, request: _Request) -> str:
        if request.status == "granted":
            return "running"
        return request.status

    def _lease_snapshot_locked(self, lease: _Lease, now: float) -> dict[str, Any]:
        return {
            "lease_id": lease.lease_id,
            "task_id": lease.request_id,
            "uid": lease.uid,
            "user": lease.user,
            "owner": lease.owner,
            "cards": list(lease.cards),
            "expires_at": lease.expires_at,
            "ttl_seconds": self._lease_ttl,
            "remaining_seconds": max(0.0, lease.deadline - now),
        }

    def _authorize_request_locked(
        self,
        request: _Request,
        *,
        uid: int | None,
        admin: bool,
    ) -> None:
        if uid is None or admin or request.uid == uid:
            return
        raise PermissionDeniedError(
            f"task {request.request_id} belongs to uid {request.uid}"
        )

    def _reaper_loop(self) -> None:
        while True:
            with self._condition:
                if self._closed:
                    return
                now = time.monotonic()
                self._maintain_locked(now)
                deadlines = [
                    lease.deadline for lease in self._leases.values()
                ] + [
                    self._requests[request_id].queue_deadline
                    for request_id in self._queue
                ]
                timeout = None
                if deadlines:
                    timeout = max(0.0, min(deadlines) - now)
                self._condition.wait(timeout=timeout)
