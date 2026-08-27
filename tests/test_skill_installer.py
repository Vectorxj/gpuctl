from __future__ import annotations

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from gpuctl.client import main
from gpuctl.skill_installer import (
    SKILL_NAME,
    SkillInstallError,
    bundled_skill_text,
    install_skill,
)


class SkillInstallerTest(unittest.TestCase):
    def test_project_skill_matches_bundled_skill(self) -> None:
        project_skill = (
            Path(__file__).resolve().parents[1]
            / ".claude"
            / "skills"
            / SKILL_NAME
            / "SKILL.md"
        )
        self.assertEqual(project_skill.read_text(encoding="utf-8"), bundled_skill_text())

    def test_skill_has_required_frontmatter_and_rules(self) -> None:
        content = bundled_skill_text()
        self.assertTrue(content.startswith("---\n"))
        self.assertIn(f"\nname: {SKILL_NAME}\n", content)
        self.assertIn("\ndescription:", content)
        self.assertIn("gpuctl --count N", content)
        self.assertIn("do not rerun the GPU command without", content)

    def test_installs_for_both_agents_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            first = install_skill(home=home)
            self.assertEqual([result.agent for result in first], ["copilot", "claude"])
            self.assertTrue(all(result.changed for result in first))

            expected = bundled_skill_text()
            for result in first:
                self.assertEqual(result.destination.read_text(encoding="utf-8"), expected)

            second = install_skill(home=home)
            self.assertTrue(all(not result.changed for result in second))

    def test_refuses_modified_skill_without_partial_install(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            claude_skill = (
                home / ".claude" / "skills" / SKILL_NAME / "SKILL.md"
            )
            claude_skill.parent.mkdir(parents=True)
            claude_skill.write_text("custom\n", encoding="utf-8")

            with self.assertRaisesRegex(SkillInstallError, "--force"):
                install_skill(home=home)

            copilot_skill = (
                home / ".copilot" / "skills" / SKILL_NAME / "SKILL.md"
            )
            self.assertFalse(copilot_skill.exists())
            self.assertEqual(claude_skill.read_text(encoding="utf-8"), "custom\n")

    def test_force_replaces_modified_skill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            destination = (
                home / ".copilot" / "skills" / SKILL_NAME / "SKILL.md"
            )
            destination.parent.mkdir(parents=True)
            destination.write_text("custom\n", encoding="utf-8")

            result = install_skill("copilot", force=True, home=home)
            self.assertTrue(result[0].changed)
            self.assertEqual(destination.read_text(encoding="utf-8"), bundled_skill_text())

    def test_install_skill_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            stdout = io.StringIO()
            with (
                mock.patch(
                    "gpuctl.skill_installer.Path.home",
                    return_value=Path(directory),
                ),
                contextlib.redirect_stdout(stdout),
            ):
                result = main(["install-skill", "--agent", "claude"])

            self.assertEqual(result, 0)
            self.assertIn("claude: installed:", stdout.getvalue())
            installed = (
                Path(directory)
                / ".claude"
                / "skills"
                / SKILL_NAME
                / "SKILL.md"
            )
            self.assertTrue(installed.is_file())
