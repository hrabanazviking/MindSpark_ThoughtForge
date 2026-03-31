"""
run_thoughtforge.py — MindSpark: ThoughtForge interactive CLI.

Usage:
  # Interactive REPL:
  python run_thoughtforge.py

  # Single query:
  python run_thoughtforge.py "What is Yggdrasil?"

  # With a GGUF model:
  python run_thoughtforge.py --model /models/phi-3-mini-q4.gguf

  # Override hardware profile:
  python run_thoughtforge.py --profile desktop_cpu

  # Debug logging:
  python run_thoughtforge.py --debug
"""

from __future__ import annotations

import argparse
import logging
import sys

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_thoughtforge",
        description="MindSpark: ThoughtForge — Memory-Enforced Cognition Engine",
    )
    p.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Single query to run (omit for interactive REPL mode)",
    )
    p.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help="Path to a GGUF model file (optional — knowledge-only mode if omitted)",
    )
    p.add_argument(
        "--profile",
        metavar="PROFILE",
        default="auto",
        help="Hardware profile: auto|phone_low|pi_zero|pi_5|desktop_cpu|desktop_gpu|server_gpu",
    )
    p.add_argument(
        "--retrieval",
        metavar="PATH",
        default=None,
        choices=["sql", "vector", "hybrid"],
        help="Retrieval path override: sql|vector|hybrid (default: auto from intent)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging",
    )
    return p


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        format="%(levelname)s [%(name)s] %(message)s",
        level=level,
        stream=sys.stderr,
    )


def _display_result(result: object, query: str) -> None:
    """Print the FinalResponseRecord in a readable format."""
    sep = "═" * 72

    print(f"\n{sep}")
    print(result.text)  # type: ignore[attr-defined]
    print(sep)

    citations = getattr(result, "citations", [])
    scores = getattr(result, "scores", None)
    enforcement_passed = getattr(result, "enforcement_passed", False)
    enforcement_notes = getattr(result, "enforcement_notes", "")
    token_count = getattr(result, "token_count", 0)

    cite_str = ", ".join(citations) if citations else "none"
    confidence = scores.composite if scores else 0.0
    quality = scores.quality_tier if scores else "—"
    enf_str = "PASS" if enforcement_passed else f"REVIEW ({enforcement_notes})"

    print(f"Citations   : {cite_str}")
    print(f"Confidence  : {confidence:.3f}  [{quality}]")
    print(f"Enforcement : {enf_str}")
    print(f"Tokens      : {token_count}")
    print()


def _run_single(core: object, query: str, retrieval_path: str | None) -> None:
    result = core.think(query, retrieval_path=retrieval_path)  # type: ignore[attr-defined]
    _display_result(result, query)


def _run_repl(core: object, retrieval_path: str | None) -> None:
    print("MindSpark: ThoughtForge")
    print("The forge is ready. Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            query = input("Forge> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThe forge grows quiet. Walk well.")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q", ":q"):
            print("The forge grows quiet. Walk well.")
            break

        try:
            result = core.think(query, retrieval_path=retrieval_path)  # type: ignore[attr-defined]
            _display_result(result, query)
        except Exception as e:
            print(f"[Forge Error] {e}", file=sys.stderr)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    _setup_logging(args.debug)

    # Import here so logging is set up first
    from thoughtforge.cognition.core import ThoughtForgeCore
    from thoughtforge.utils.logging_setup import setup_logging

    if not args.debug:
        setup_logging(config={"logging": {"level": "WARNING"}})

    model_path = Path(args.model) if args.model else None

    core = ThoughtForgeCore(model_path=model_path)

    if args.query:
        _run_single(core, args.query, args.retrieval)
    else:
        _run_repl(core, args.retrieval)


if __name__ == "__main__":
    main()
