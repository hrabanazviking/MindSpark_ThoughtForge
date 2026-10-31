"""
forge_doctor.py — MindSpark: ThoughtForge System Diagnostics

Usage:
  python forge_doctor.py              # Run all checks
  python forge_doctor.py --fix        # Run checks + attempt self-healing
  python forge_doctor.py --json       # Machine-readable JSON output
  python forge_doctor.py --verbose    # Show all checks including passing ones
  python forge_doctor.py --audit      # Also emit the campaign audit headline numbers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure src/ is on the path when run from the project root
_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="forge_doctor",
        description="MindSpark: ThoughtForge — System Diagnostics",
    )
    p.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to self-heal any detected issues",
    )
    p.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Output results as JSON",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Show all checks including passing ones (default shows issues only in short mode)",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Also emit the Part I audit headline numbers measured live from the tree",
    )
    return p


def audit_headlines(root: Path | None = None) -> dict:
    """Headline numbers of the campaign audit, measured live from the tree.

    Only numbers that can be counted cheaply and truthfully from the
    working tree itself are emitted.  Deliberately NOT emitted (see
    docs/audit/TF-013_DOCTORS_REPORT.md): test pass/fail counts (those
    come from running the suite, TF-004/TF-014), the data/ canon census
    totals (a sparse checkout would under-count them, TF-007), config
    live/dead key classification (requires code tracing, TF-009), and
    benchmark scores (require running the benchmarks, TF-010).
    """
    root = root or _HERE
    src_files = sorted((root / "src" / "thoughtforge").rglob("*.py"))
    test_files = sorted((root / "tests").glob("test_*.py"))
    profiles = sorted(p.stem for p in (root / "hardware_profiles").glob("*.json"))
    configs = sorted(p.name for p in (root / "configs").glob("*.yaml"))
    bench = sorted(
        p.name for p in (root / "benchmarks").glob("*.py") if p.name != "__init__.py"
    )
    audit_docs = sorted((root / "docs" / "audit").glob("TF-*.md"))
    return {
        "source_py_files": len(src_files),
        "source_bytes": sum(p.stat().st_size for p in src_files),
        "test_files": len(test_files),
        "hardware_profiles": profiles,
        "hardware_profile_count": len(profiles),
        "config_files": configs,
        "benchmark_modules": bench,
        "audit_docs": len(audit_docs),
        "knowledge_db_present": (root / "data" / "knowledge.db").exists(),
    }


def _print_audit_text(headlines: dict) -> None:
    print("\n[Audit Headlines] (measured live from this tree)")
    print(f"  Source: {headlines['source_py_files']} Python files, "
          f"{headlines['source_bytes']:,} bytes under src/thoughtforge")
    print(f"  Tests: {headlines['test_files']} test files under tests/ "
          "(pass/fail counts come from the suite runner, not the doctor)")
    print(f"  Hardware profiles: {headlines['hardware_profile_count']} "
          f"({', '.join(headlines['hardware_profiles'])})")
    print(f"  Config files: {', '.join(headlines['config_files'])}")
    print(f"  Benchmark modules: {', '.join(headlines['benchmark_modules'])}")
    print(f"  Audit docs filed: {headlines['audit_docs']} under docs/audit")
    print(f"  Knowledge DB present: {'yes' if headlines['knowledge_db_present'] else 'no'}")


def _run_checks() -> list:
    from thoughtforge.utils.health import HealthChecker
    return HealthChecker().check_all()


def _run_heal() -> dict[str, bool]:
    from thoughtforge.utils.self_heal import SelfHealer
    return SelfHealer().heal_all()


def _print_text_report(results: list, heal_results: dict[str, bool] | None, verbose: bool) -> None:
    print()
    print("MindSpark: ThoughtForge — System Diagnostics")
    print("=" * 45)

    # Group by component
    sections: dict[str, list] = {}
    for r in results:
        sections.setdefault(r.component, []).append(r)

    for component, comp_results in sections.items():
        any_shown = False
        section_lines = []
        for r in comp_results:
            if not verbose and r.ok:
                continue
            icon = "✓" if r.ok else ("!" if r.status == "degraded" else "✗")
            section_lines.append(f"  {icon} {r.message}")
            if not r.ok and r.fix_hint:
                section_lines.append(f"    → {r.fix_hint}")
            any_shown = True

        if verbose:
            # Show all checks including passing
            section_lines = []
            for r in comp_results:
                icon = "✓" if r.ok else ("!" if r.status == "degraded" else "✗")
                section_lines.append(f"  {icon} {r.message}")
                if not r.ok and r.fix_hint:
                    section_lines.append(f"    → {r.fix_hint}")
            print(f"\n[{component}]")
            for line in section_lines:
                print(line)
        elif section_lines:
            print(f"\n[{component}]")
            for line in section_lines:
                print(line)

    # Self-heal results
    if heal_results is not None:
        print("\n[Self-Heal]")
        any_issues = False
        for component, success in heal_results.items():
            if not success:
                print(f"  ✗ {component}: healing failed — check logs for details")
                any_issues = True
        if not any_issues:
            # Were there issues to fix?
            had_issues = any(not r.ok for r in results)
            if had_issues:
                print("  ✓ All detected issues repaired successfully")
            else:
                print("  No issues found. Nothing to repair.")

    # Summary
    failed = [r for r in results if r.failed]
    degraded = [r for r in results if r.status == "degraded"]
    print()
    if not failed and not degraded:
        print("All systems operational. Run: python run_thoughtforge.py")
    elif not failed:
        print(
            f"{len(degraded)} degraded component(s) (non-critical). "
            "Run: python run_thoughtforge.py"
        )
    else:
        print(
            f"{len(failed)} issue(s) found."
            + (" Run: python forge_doctor.py --fix" if heal_results is None else "")
        )
    print()


def _print_json_report(
    results: list,
    heal_results: dict[str, bool] | None,
    headlines: dict | None = None,
) -> None:
    output = {
        "checks": [
            {
                "component": r.component,
                "status": r.status,
                "message": r.message,
                "fix_hint": r.fix_hint,
            }
            for r in results
        ],
        "summary": {
            "total": len(results),
            "ok": sum(1 for r in results if r.ok),
            "degraded": sum(1 for r in results if r.status == "degraded"),
            "failed": sum(1 for r in results if r.failed),
        },
    }
    if heal_results is not None:
        output["heal_results"] = heal_results
    if headlines is not None:
        output["audit_headlines"] = headlines
    print(json.dumps(output, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Run health checks
    try:
        results = _run_checks()
    except Exception as exc:
        print(f"Error running health checks: {exc}", file=sys.stderr)
        return 2

    # Run self-healer if requested
    heal_results: dict[str, bool] | None = None
    if args.fix:
        had_issues = any(not r.ok for r in results)
        try:
            heal_results = _run_heal()
        except Exception as exc:
            print(f"Error during self-healing: {exc}", file=sys.stderr)
            if args.as_json:
                _print_json_report(results, None)
            else:
                _print_text_report(results, None, args.verbose)
            return 2
        # Re-run checks after healing to reflect updated state
        try:
            results = _run_checks()
        except Exception:
            pass

    headlines = audit_headlines() if args.audit else None
    if args.as_json:
        _print_json_report(results, heal_results, headlines)
    else:
        _print_text_report(results, heal_results, args.verbose)
        if headlines is not None:
            _print_audit_text(headlines)

    failed = sum(1 for r in results if r.failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
