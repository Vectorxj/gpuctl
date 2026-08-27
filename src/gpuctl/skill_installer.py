from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import os
from pathlib import Path


SKILL_NAME = "gpuctl-gpu"
_AGENT_DIRECTORIES = {
    "copilot": Path(".copilot/skills"),
    "claude": Path(".claude/skills"),
}


class SkillInstallError(Exception):
    pass


@dataclass(frozen=True)
class InstallResult:
    agent: str
    destination: Path
    changed: bool


def bundled_skill_text() -> str:
    resource = (
        files("gpuctl")
        .joinpath("skills")
        .joinpath(SKILL_NAME)
        .joinpath("SKILL.md")
    )
    try:
        return resource.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as error:
        raise SkillInstallError("the installed package does not contain the GPU skill") from error


def install_skill(
    agent: str = "all",
    *,
    force: bool = False,
    home: Path | None = None,
) -> list[InstallResult]:
    if agent != "all" and agent not in _AGENT_DIRECTORIES:
        raise SkillInstallError(f"unsupported agent: {agent}")

    home_directory = home if home is not None else Path.home()
    selected_agents = (
        tuple(_AGENT_DIRECTORIES) if agent == "all" else (agent,)
    )
    content = bundled_skill_text()
    plans: list[tuple[str, Path, bool]] = []

    for agent_name in selected_agents:
        destination = (
            home_directory
            / _AGENT_DIRECTORIES[agent_name]
            / SKILL_NAME
            / "SKILL.md"
        )
        changed = True
        if destination.exists():
            if not destination.is_file():
                raise SkillInstallError(
                    f"skill destination is not a file: {destination}"
                )
            try:
                existing = destination.read_text(encoding="utf-8")
            except OSError as error:
                raise SkillInstallError(
                    f"cannot read existing skill at {destination}: {error}"
                ) from error
            if existing == content:
                changed = False
            elif not force:
                raise SkillInstallError(
                    f"refusing to overwrite modified skill at {destination}; "
                    "use --force to replace it"
                )
        plans.append((agent_name, destination, changed))

    results = []
    for agent_name, destination, changed in plans:
        if changed:
            _write_skill(destination, content)
        results.append(
            InstallResult(
                agent=agent_name,
                destination=destination,
                changed=changed,
            )
        )
    return results


def _write_skill(destination: Path, content: str) -> None:
    temporary: Path | None = None
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, destination)
    except OSError as error:
        raise SkillInstallError(
            f"cannot install skill at {destination}: {error}"
        ) from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
