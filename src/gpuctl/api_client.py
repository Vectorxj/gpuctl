from __future__ import annotations

from http.client import HTTPConnection, HTTPException
import json
import socket
from typing import Any
from urllib.parse import quote

from . import __version__


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


class TransportError(Exception):
    pass


class _UnixHTTPConnection(HTTPConnection):
    def __init__(self, socket_path: str, *, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            connection.settimeout(self.timeout)
            connection.connect(self._socket_path)
        except (OSError, socket.timeout):
            connection.close()
            raise
        self.sock = connection


class APIClient:
    def __init__(
        self,
        socket_path: str,
        *,
        timeout: float = 10.0,
    ) -> None:
        if not socket_path:
            raise ValueError("socket path must not be empty")
        if timeout <= 0:
            raise ValueError("request timeout must be positive")
        self._socket_path = socket_path
        self._timeout = timeout

    def create_request(
        self,
        *,
        owner: str,
        cards: tuple[str, ...] | None,
        count: int | None,
        command: str,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "owner": owner,
            "command": command,
        }
        if cards is not None:
            payload["cards"] = list(cards)
        else:
            payload["count"] = count
        return self._request("POST", "/v1/requests", payload)

    def get_request(self, request_id: str) -> dict[str, Any]:
        return self._request("GET", f"/v1/requests/{quote(request_id, safe='')}")

    def cancel_request(self, request_id: str) -> None:
        self._request("DELETE", f"/v1/requests/{quote(request_id, safe='')}")

    def renew_lease(
        self, lease_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        return self._request(
            "PUT",
            f"/v1/leases/{quote(lease_id, safe='')}",
            {},
            timeout=timeout,
        )

    def release_lease(self, lease_id: str) -> None:
        self._request("DELETE", f"/v1/leases/{quote(lease_id, safe='')}")

    def cancel_task(self, task_id: str) -> None:
        self._request("DELETE", f"/v1/tasks/{quote(task_id, safe='')}")

    def tasks(self, *, include_all: bool = False) -> dict[str, Any]:
        suffix = "?all=true" if include_all else ""
        return self._request("GET", f"/v1/tasks{suffix}")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "/v1/status")

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = None
        headers = {
            "Accept": "application/json",
            "User-Agent": f"gpuctl/{__version__}",
        }
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"

        connection = _UnixHTTPConnection(
            self._socket_path,
            timeout=timeout or self._timeout,
        )
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
            if response.status >= 400:
                code = "http_error"
                message = f"daemon returned HTTP {response.status}"
                if response_body:
                    try:
                        value = json.loads(response_body.decode("utf-8"))
                        details = value.get("error", {})
                        if isinstance(details, dict):
                            code = str(details.get("code", code))
                            message = str(details.get("message", message))
                    except (json.JSONDecodeError, UnicodeDecodeError, AttributeError):
                        pass
                raise APIError(response.status, code, message)
            if not response_body:
                return {}
            value = json.loads(response_body.decode("utf-8"))
            if not isinstance(value, dict):
                raise TransportError("daemon returned a non-object JSON response")
            return value
        except APIError:
            raise
        except (HTTPException, OSError, socket.timeout, TimeoutError) as error:
            raise TransportError(
                f"cannot reach gpuctld at {self._socket_path}: {error}"
            ) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise TransportError("daemon returned invalid JSON") from error
        finally:
            connection.close()
