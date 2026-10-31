"""TF-013 — forge_doctor --audit headline numbers.

The doctor must emit the Part I audit's headline numbers by itself,
measured live from the tree — never quoted from the audit docs.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _load_doctor():
    spec = importlib.util.spec_from_file_location(
        "forge_doctor", REPO_ROOT / "forge_doctor.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


doctor = _load_doctor()


class TestAuditHeadlines:
    def test_source_census_matches_independent_recount(self):
        h = doctor.audit_headlines()
        files = sorted((REPO_ROOT / "src" / "thoughtforge").rglob("*.py"))
        assert h["source_py_files"] == len(files)
        assert h["source_bytes"] == sum(p.stat().st_size for p in files)
        assert h["source_py_files"] > 0

    def test_profiles_and_configs_named(self):
        h = doctor.audit_headlines()
        assert h["hardware_profile_count"] == 6
        assert "desktop_cpu" in h["hardware_profiles"]
        assert "pi_zero" in h["hardware_profiles"]
        assert "user_config.yaml" in h["config_files"]

    def test_benchmarks_and_audit_docs_counted(self):
        h = doctor.audit_headlines()
        assert "persona_consistency.py" in h["benchmark_modules"]
        assert "benchmark_profiles.py" in h["benchmark_modules"]
        assert h["audit_docs"] >= 11  # TF-002..TF-012 docs (TF-001 filed none)
        assert h["test_files"] >= 13

    def test_root_parameter_is_respected(self, tmp_path):
        (tmp_path / "src" / "thoughtforge").mkdir(parents=True)
        (tmp_path / "src" / "thoughtforge" / "a.py").write_text("x = 1\n")
        for d in ("tests", "hardware_profiles", "configs", "benchmarks",
                  "docs/audit", "data"):
            (tmp_path / d).mkdir(parents=True, exist_ok=True)
        h = doctor.audit_headlines(root=tmp_path)
        assert h["source_py_files"] == 1
        assert h["knowledge_db_present"] is False


class TestAuditCli:
    def test_json_output_carries_headlines(self, capsys):
        rc = doctor.main(["--audit", "--json"])
        assert rc in (0, 1)  # 1 only while D-ENV-1 env failure stands
        out = json.loads(capsys.readouterr().out)
        assert "audit_headlines" in out
        assert out["audit_headlines"]["hardware_profile_count"] == 6
        assert "summary" in out and "checks" in out

    def test_no_audit_key_without_flag(self, capsys):
        doctor.main(["--json"])
        out = json.loads(capsys.readouterr().out)
        assert "audit_headlines" not in out

    def test_text_output_has_audit_section(self, capsys):
        doctor.main(["--audit"])
        out = capsys.readouterr().out
        assert "[Audit Headlines]" in out
