#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CARDS=""
RUN_TESTS=true
INSTALL_SKILL=true
FORCE_RESTART=false
BUILD_ONLY=false
BUILD_DIR=""

usage() {
    cat <<'EOF'
Usage: ./install.sh [OPTIONS]

Build and install or upgrade gpuctl for every user on this machine.
Run this script as a regular user; it invokes sudo for system changes.

Options:
  --cards IDS        Set managed GPU IDs, for example 0,1,2,3.
                     Without this option, preserve existing configuration or
                     detect /dev/nvidiaN devices during the first install.
  --skip-tests       Build and install without running the test suite.
  --skip-skill       Do not refresh Copilot and Claude skills for this user.
  --force-restart    Restart even with tasks, leaving running workloads
                     untracked. Use only during coordinated maintenance.
  --build-only       Run tests and build dist/gpuctl-*.whl without installing.
  -h, --help         Show this help.

Environment:
  PYTHON              System Python 3.10+ interpreter (default: python3).
EOF
}

log() {
    printf '\n==> %s\n' "$*"
}

die() {
    printf 'install.sh: %s\n' "$*" >&2
    exit 1
}

cleanup() {
    if [[ -n "${BUILD_DIR:-}" &&
          -d "$BUILD_DIR" &&
          "$(basename -- "$BUILD_DIR")" == gpuctl-install.* ]]; then
        rm -rf -- "$BUILD_DIR"
    fi
}

detect_cards() {
    local device card
    local -a cards=()

    shopt -s nullglob
    for device in /dev/nvidia[0-9]*; do
        [[ -c "$device" ]] || continue
        card="${device##*/nvidia}"
        cards+=("$card")
    done
    shopt -u nullglob

    ((${#cards[@]} > 0)) || return 1
    printf '%s\n' "${cards[@]}" | sort -n | paste -sd, -
}

validate_cards() {
    local value="$1" card
    local -a cards=()
    local -A seen=()

    [[ "$value" =~ ^[0-9]+(,[0-9]+)*$ ]] || return 1
    IFS=',' read -r -a cards <<<"$value"
    for card in "${cards[@]}"; do
        [[ -z "${seen[$card]+present}" ]] || return 1
        seen["$card"]=1
    done
}

while (($# > 0)); do
    case "$1" in
        --cards)
            (($# >= 2)) || die "--cards requires a comma-separated value"
            CARDS="$2"
            shift 2
            ;;
        --cards=*)
            CARDS="${1#*=}"
            shift
            ;;
        --skip-tests)
            RUN_TESTS=false
            shift
            ;;
        --skip-skill)
            INSTALL_SKILL=false
            shift
            ;;
        --force-restart)
            FORCE_RESTART=true
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

command -v "$PYTHON_BIN" >/dev/null 2>&1 ||
    die "Python interpreter not found: $PYTHON_BIN"
PYTHON_BIN="$(command -v "$PYTHON_BIN")"

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(sys.version_info < (3, 10))'; then
    die "Python 3.10 or later is required"
fi
if [[ -n "$CARDS" ]]; then
    validate_cards "$CARDS" ||
        die "--cards must contain unique numeric IDs such as 0,1,2,3"
fi

trap cleanup EXIT

if [[ "$RUN_TESTS" == true ]]; then
    log "Running tests"
    (
        cd "$ROOT_DIR"
        PYTHONPATH=src "$PYTHON_BIN" -m unittest discover -s tests -v
    )
fi

log "Building wheel"
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/gpuctl-install.XXXXXX")"
"$PYTHON_BIN" -m pip wheel "$ROOT_DIR" --no-deps -w "$BUILD_DIR"

mapfile -t WHEELS < <(
    find "$BUILD_DIR" -maxdepth 1 -type f -name 'gpuctl-*.whl' -print
)
((${#WHEELS[@]} == 1)) ||
    die "expected exactly one gpuctl wheel, found ${#WHEELS[@]}"

mkdir -p "$ROOT_DIR/dist"
WHEEL_NAME="$(basename -- "${WHEELS[0]}")"
WHEEL_PATH="$ROOT_DIR/dist/$WHEEL_NAME"
install -m 0644 "${WHEELS[0]}" "$WHEEL_PATH"
log "Built $WHEEL_PATH"

if [[ "$BUILD_ONLY" == true ]]; then
    exit 0
fi

((EUID != 0)) ||
    die "run this script as a regular user, not with sudo"
command -v sudo >/dev/null 2>&1 || die "sudo is required"
command -v systemctl >/dev/null 2>&1 || die "systemctl is required"
[[ "$PYTHON_BIN" != "$HOME/"* ]] ||
    die "PYTHON must point to a system interpreter outside your home directory"
"$PYTHON_BIN" -m venv --help >/dev/null 2>&1 ||
    die "Python venv is unavailable; on Debian/Ubuntu run: sudo apt-get install python3-venv"

WRITE_CARDS=false
if [[ -n "$CARDS" ]]; then
    WRITE_CARDS=true
elif [[ -f /etc/gpuctl/gpuctld.env ]]; then
    log "Preserving /etc/gpuctl/gpuctld.env"
else
    CARDS="$(detect_cards)" ||
        die "no /dev/nvidiaN devices found; specify managed IDs with --cards"
    WRITE_CARDS=true
    log "Detected GPU cards: $CARDS"
fi

if systemctl is-active --quiet gpuctld; then
    log "Checking for active gpuctl tasks"
    if ! STATUS_JSON="$(
        PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m gpuctl status --json
    )"; then
        die "gpuctld is active but its task state could not be read"
    fi
    if ! TASK_COUNT="$(
        printf '%s' "$STATUS_JSON" |
            "$PYTHON_BIN" -c 'import json, sys; print(len(json.load(sys.stdin).get("tasks", [])))'
    )"; then
        die "gpuctld returned an invalid status response"
    fi
    if ((TASK_COUNT > 0)) && [[ "$FORCE_RESTART" != true ]]; then
        PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" -m gpuctl status >&2 || true
        die "refusing to restart gpuctld with $TASK_COUNT active or queued task(s); retry when idle or use --force-restart"
    fi
fi

log "Requesting administrator access"
sudo -v

log "Installing $WHEEL_NAME into /opt/gpuctl"
if ! sudo "$PYTHON_BIN" -m venv /opt/gpuctl; then
    die "failed to create /opt/gpuctl; on Debian/Ubuntu install python3-venv"
fi
sudo /opt/gpuctl/bin/python -m pip install \
    --force-reinstall --no-deps "$WHEEL_PATH"
sudo ln -sfn /opt/gpuctl/bin/gpuctl /usr/local/bin/gpuctl
sudo ln -sfn /opt/gpuctl/bin/gpuctld /usr/local/bin/gpuctld

sudo install -d -m 0755 /etc/gpuctl
if [[ "$WRITE_CARDS" == true ]]; then
    printf 'GPUCTL_CARDS=%s\n' "$CARDS" >"$BUILD_DIR/gpuctld.env"
    sudo install -m 0644 "$BUILD_DIR/gpuctld.env" /etc/gpuctl/gpuctld.env
fi

sudo install -m 0644 \
    "$ROOT_DIR/contrib/gpuctld.service" \
    /etc/systemd/system/gpuctld.service
sudo systemctl daemon-reload
sudo systemctl enable gpuctld >/dev/null

log "Restarting gpuctld"
if ! sudo systemctl restart gpuctld; then
    sudo systemctl status --no-pager --full gpuctld || true
    die "gpuctld failed to restart"
fi

READY=false
for _ in {1..40}; do
    if /usr/local/bin/gpuctl status >/dev/null 2>&1; then
        READY=true
        break
    fi
    sleep 0.25
done
if [[ "$READY" != true ]]; then
    sudo systemctl status --no-pager --full gpuctld || true
    die "gpuctld started but did not become ready"
fi

RESOLVED_GPUCTL="$(type -P gpuctl || true)"
USER_SITE="$("$PYTHON_BIN" -m site --user-site)"
USER_PACKAGE_LOCATION="$(
    { "$PYTHON_BIN" -m pip show gpuctl 2>/dev/null || true; } |
        sed -n 's/^Location: //p'
)"
SYNCED_USER_INSTALL=false
if [[ "$RESOLVED_GPUCTL" == "$HOME/.local/bin/gpuctl" &&
      "$USER_PACKAGE_LOCATION" == "$USER_SITE" ]]; then
    log "Refreshing the existing user-level gpuctl installation"
    "$PYTHON_BIN" -m pip install \
        --user --force-reinstall --no-deps "$WHEEL_PATH"
    SYNCED_USER_INSTALL=true
fi

if [[ "$INSTALL_SKILL" == true ]]; then
    log "Refreshing Copilot and Claude skills for $(id -un)"
    /usr/local/bin/gpuctl install-skill --force
fi

log "Installed gpuctl $(/usr/local/bin/gpuctl --version)"
/usr/local/bin/gpuctl status

if [[ -n "$RESOLVED_GPUCTL" &&
      "$RESOLVED_GPUCTL" != /usr/local/bin/gpuctl &&
      "$SYNCED_USER_INSTALL" != true ]]; then
    printf '\nWarning: %s appears before /usr/local/bin/gpuctl in PATH.\n' \
        "$RESOLVED_GPUCTL" >&2
    printf 'Remove or update that executable to avoid using a stale client.\n' >&2
fi

printf '\nOther users can now run gpuctl directly. Each user should run\n'
printf '`gpuctl install-skill` once to install personal agent skills.\n'
