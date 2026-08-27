from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
import json
import logging
import os
from pathlib import Path
import pwd
import socket
from socketserver import ThreadingMixIn, UnixStreamServer
import stat
import struct
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .manager import (
    ClosedError,
    LeaseManager,
    NotFoundError,
    PermissionDeniedError,
    TaskCanceledError,
    ValidationError,
)


MAX_BODY_BYTES = 64 * 1024


@dataclass(frozen=True)
class Caller:
    pid: int
    uid: int
    gid: int
    user: str

    @property
    def is_admin(self) -> bool:
        return self.uid == 0


class GPUUnixHTTPServer(ThreadingMixIn, UnixStreamServer):
    daemon_threads = True
    block_on_close = False

    def __init__(
        self,
        socket_path: str,
        manager: LeaseManager,
        *,
        socket_mode: int = 0o666,
        logger: logging.Logger | None = None,
    ) -> None:
        if not hasattr(socket, "SO_PEERCRED"):
            raise OSError("gpuctld requires Linux SO_PEERCRED support")
        self.manager = manager
        self.socket_path = socket_path
        self.socket_mode = socket_mode
        self.logger = logger or logging.getLogger("gpuctld.http")
        self._socket_inode: int | None = None
        self._prepare_socket_path()
        try:
            super().__init__(socket_path, GPURequestHandler)
            self._socket_inode = os.lstat(socket_path).st_ino
            os.chmod(socket_path, socket_mode)
        except OSError:
            self._remove_owned_socket()
            raise

    def caller_for(self, connection: socket.socket) -> Caller:
        credential_size = struct.calcsize("3i")
        raw = connection.getsockopt(
            socket.SOL_SOCKET,
            socket.SO_PEERCRED,
            credential_size,
        )
        pid, uid, gid = struct.unpack("3i", raw)
        try:
            user = pwd.getpwuid(uid).pw_name
        except KeyError:
            user = str(uid)
        return Caller(pid=pid, uid=uid, gid=gid, user=user)

    def server_close(self) -> None:
        super().server_close()
        self._remove_owned_socket()

    def _prepare_socket_path(self) -> None:
        path = Path(self.socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not os.path.lexists(path):
            return
        metadata = os.lstat(path)
        if not stat.S_ISSOCK(metadata.st_mode):
            raise OSError(f"refusing to replace non-socket path: {path}")

        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            probe.settimeout(0.2)
            probe.connect(self.socket_path)
        except ConnectionRefusedError:
            path.unlink()
        except OSError as error:
            raise OSError(f"cannot inspect existing socket {path}: {error}") from error
        else:
            raise OSError(f"socket is already in use: {path}")
        finally:
            probe.close()

    def _remove_owned_socket(self) -> None:
        if self._socket_inode is None:
            return
        try:
            metadata = os.lstat(self.socket_path)
            if stat.S_ISSOCK(metadata.st_mode) and metadata.st_ino == self._socket_inode:
                os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        finally:
            self._socket_inode = None


class GPURequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "gpuctld/0.1"
    sys_version = ""

    @property
    def gpu_server(self) -> GPUUnixHTTPServer:
        return self.server  # type: ignore[return-value]

    @property
    def caller(self) -> Caller:
        caller = getattr(self, "_caller", None)
        if caller is None:
            caller = self.gpu_server.caller_for(self.connection)
            self._caller = caller
        return caller

    def do_GET(self) -> None:
        self._handle_request("GET")

    def do_POST(self) -> None:
        self._handle_request("POST")

    def do_PUT(self) -> None:
        self._handle_request("PUT")

    def do_DELETE(self) -> None:
        self._handle_request("DELETE")

    def log_message(self, message: str, *args: Any) -> None:
        self.gpu_server.logger.debug(
            "uid=%s pid=%s - %s",
            self.caller.uid,
            self.caller.pid,
            message % args,
        )

    def _handle_request(self, method: str) -> None:
        try:
            self._dispatch(method)
        except ValidationError as error:
            self._send_error(400, "invalid_request", str(error))
        except PermissionDeniedError as error:
            self._send_error(403, "forbidden", str(error))
        except TaskCanceledError as error:
            self._send_error(409, "task_canceled", str(error))
        except NotFoundError as error:
            self._send_error(404, "not_found", str(error))
        except ClosedError as error:
            self._send_error(503, "shutting_down", str(error))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_error(400, "invalid_json", "request body must be valid JSON")
        except ValueError as error:
            self._send_error(400, "invalid_request", str(error))
        except (BrokenPipeError, ConnectionResetError):
            self.gpu_server.logger.debug("client disconnected before response")
        except Exception:
            self.gpu_server.logger.exception("unhandled HTTP API error")
            self._send_error(500, "internal_error", "internal server error")

    def _dispatch(self, method: str) -> None:
        parsed = urlsplit(self.path)
        path = parsed.path
        if path == "/healthz" and method == "GET":
            self._send_json(200, {"status": "ok"})
            return

        parts = [unquote(part) for part in path.strip("/").split("/") if part]
        manager = self.gpu_server.manager
        caller = self.caller

        if method == "GET" and parts == ["v1", "status"]:
            self._send_json(200, manager.status())
            return

        if method == "GET" and parts == ["v1", "tasks"]:
            query = parse_qs(parsed.query)
            include_all = query.get("all") == ["true"]
            if include_all and not caller.is_admin:
                raise PermissionDeniedError("only uid 0 can list all users' tasks")
            self._send_json(
                200,
                manager.tasks(uid=caller.uid, include_all=include_all),
            )
            return

        if method == "POST" and parts == ["v1", "requests"]:
            body = self._read_json()
            state = manager.create_request(
                owner=body.get("owner"),
                cards=body.get("cards"),
                count=body.get("count"),
                uid=caller.uid,
                user=caller.user,
                client_pid=caller.pid,
                command=body.get("command"),
                reservation_seconds=body.get("reservation_seconds"),
            )
            self._send_json(201, state)
            return

        if len(parts) == 3 and parts[:2] == ["v1", "requests"]:
            request_id = parts[2]
            if method == "GET":
                self._send_json(
                    200,
                    manager.get_request(
                        request_id,
                        uid=caller.uid,
                        admin=caller.is_admin,
                    ),
                )
                return
            if method == "DELETE":
                if not manager.cancel_request(
                    request_id,
                    uid=caller.uid,
                    admin=caller.is_admin,
                ):
                    raise NotFoundError("request does not exist or has expired")
                self._send_empty(204)
                return

        if len(parts) == 3 and parts[:2] == ["v1", "leases"]:
            lease_id = parts[2]
            if method == "PUT":
                self._read_optional_json()
                self._send_json(
                    200,
                    manager.renew_lease(
                        lease_id,
                        uid=caller.uid,
                        admin=caller.is_admin,
                    ),
                )
                return
            if method == "DELETE":
                manager.release_lease(
                    lease_id,
                    uid=caller.uid,
                    admin=caller.is_admin,
                )
                self._send_empty(204)
                return

        if (
            method == "DELETE"
            and len(parts) == 3
            and parts[:2] == ["v1", "tasks"]
        ):
            task_id = parts[2]
            if not manager.request_task_cancel(
                task_id,
                uid=caller.uid,
                admin=caller.is_admin,
            ):
                raise NotFoundError("task does not exist")
            self._send_empty(204)
            return

        self._send_error(404, "not_found", "endpoint not found")

    def _read_json(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(length_header)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError(f"request body must not exceed {MAX_BODY_BYTES} bytes")
        payload = self.rfile.read(length)
        value = json.loads(payload.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _read_optional_json(self) -> None:
        length_header = self.headers.get("Content-Length")
        if length_header is None or length_header == "0":
            return
        self._read_json()

    def _send_error(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": message}})

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self.send_response(status)
        self._finish_json(payload)

    def _finish_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_empty(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()
