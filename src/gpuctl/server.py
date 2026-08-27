from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
from typing import Sequence

from . import __version__
from .common import DEFAULT_SOCKET_PATH, cards_argument, duration_argument
from .http_api import GPUUnixHTTPServer
from .manager import LeaseManager


class _ShutdownRequested(Exception):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpuctld",
        description="Run the gpuctl GPU lease server.",
    )
    parser.add_argument(
        "--cards",
        required=True,
        type=cards_argument,
        metavar="ID,ID,...",
        help="card IDs managed by this server, in idle-allocation order",
    )
    parser.add_argument(
        "--socket",
        default=os.environ.get("GPUCTL_SOCKET", DEFAULT_SOCKET_PATH),
        help=f"Unix socket path (default: {DEFAULT_SOCKET_PATH})",
    )
    parser.add_argument(
        "--lease-ttl",
        type=duration_argument,
        default=30.0,
        metavar="DURATION",
        help="lease lifetime without renewal (default: 30s)",
    )
    parser.add_argument(
        "--queue-ttl",
        type=duration_argument,
        default=30.0,
        metavar="DURATION",
        help="queued request lifetime without polling (default: 30s)",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.lease_ttl < 3.0:
        parser.error("--lease-ttl must be at least 3s")
    if args.queue_ttl < 3.0:
        parser.error("--queue-ttl must be at least 3s")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("gpuctld")
    manager = LeaseManager(
        args.cards,
        lease_ttl=args.lease_ttl,
        queue_ttl=args.queue_ttl,
    )
    try:
        server = GPUUnixHTTPServer(
            args.socket,
            manager,
            logger=logging.getLogger("gpuctld.http"),
        )
    except OSError as error:
        manager.close()
        parser.exit(1, f"gpuctld: cannot create socket {args.socket}: {error}\n")

    old_handlers: dict[int, signal.Handlers] = {}

    def request_shutdown(signum: int, _frame: object) -> None:
        raise _ShutdownRequested(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, request_shutdown)

    logger.info(
        "listening on %s; cards=%s; lease_ttl=%ss; queue_ttl=%ss",
        args.socket,
        ",".join(args.cards),
        args.lease_ttl,
        args.queue_ttl,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except (_ShutdownRequested, KeyboardInterrupt):
        logger.info("shutting down")
    finally:
        server.server_close()
        manager.close()
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
