from __future__ import annotations

import argparse
import re


DEFAULT_SOCKET_PATH = "/run/gpuctl/gpuctld.sock"


_DURATION_RE = re.compile(r"^(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>ms|s|m|h)?$")
_DURATION_MULTIPLIERS = {
    None: 1.0,
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}


def parse_duration(value: str) -> float:
    match = _DURATION_RE.fullmatch(value.strip().lower())
    if match is None:
        raise ValueError("expected a duration such as 30s, 5m, or 1h")
    amount = float(match.group("value"))
    return amount * _DURATION_MULTIPLIERS[match.group("unit")]


def duration_argument(value: str) -> float:
    try:
        return parse_duration(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_cards(value: str) -> tuple[str, ...]:
    raw_cards = value.split(",")
    cards = tuple(card.strip() for card in raw_cards)
    if not cards or any(not card for card in cards):
        raise ValueError("cards must be a comma-separated list of non-empty IDs")
    if len(set(cards)) != len(cards):
        raise ValueError("cards must not contain duplicate IDs")
    return cards


def cards_argument(value: str) -> tuple[str, ...]:
    try:
        return parse_cards(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
