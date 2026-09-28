#!/usr/bin/env python3
"""Build a tiny self-contained benchmark for a quick end-to-end run.

It creates a small git repository ("unitconv", a unit-conversion CLI) with a few house
conventions that are easy to miss from the task description alone:

  * every converter lives in unitconv/converters/<kind>.py, declares FACTORS relative to a
    base unit and is registered with @register("<kind>");
  * the module must be imported in unitconv/converters/__init__.py (alphabetical order),
    otherwise it is never registered;
  * unknown units raise errors.unknown_unit(...) instead of a bare ValueError/KeyError;
  * tests go to tests/test_<kind>.py and use tests/helpers.assert_close;
  * README.md gets a table row and CHANGELOG.md an "Unreleased" entry.

History: a base commit, two learning commits (weight, volume converters) and one held-out
test commit (speed converter). Snapshots are `git archive <sha>~1`, oracle diffs are
`git diff <sha>~1 <sha>`, exactly as for the real benchmark.

Usage:
    python examples/toy/build_toy_benchmark.py            # writes examples/toy/build/
    python workflow/run_sequential.py --input-file examples/toy/build/benchmark_tasks.jsonl \\
        --signature toy --run-baseline
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
BUILD = HERE / "build"
REPO = BUILD / "repo"

GIT_ENV_FLAGS = ["-c", "user.name=Toy Maintainer", "-c", "user.email=toy@example.com",
                 "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "-c", "init.defaultBranch=main"]


# Fixed dates make the commit SHAs (and hence task ids) identical on every build.
GIT_ENV = {**os.environ, "GIT_AUTHOR_DATE": "2025-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2025-01-01T00:00:00Z"}


def git(*args: str) -> str:
    out = subprocess.run(["git", *GIT_ENV_FLAGS, *args], cwd=REPO, capture_output=True, check=True, env=GIT_ENV)
    return out.stdout.decode()


def write(files: dict[str, str]):
    for rel, content in files.items():
        path = REPO / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.lstrip("\n"), encoding="utf-8")


def converter_module(kind: str, base: str, factors: dict[str, float], doc: str) -> str:
    rows = "\n".join(f'    "{u}": {f!r},' for u, f in factors.items())
    return f'''
"""{doc}"""

from unitconv.errors import unknown_unit
from unitconv.registry import register

# Multiplier to convert one unit into the base unit ({base}).
FACTORS = {{
{rows}
}}


@register("{kind}")
def convert(value: float, src: str, dst: str) -> float:
    for unit in (src, dst):
        if unit not in FACTORS:
            raise unknown_unit("{kind}", unit, FACTORS)
    return value * FACTORS[src] / FACTORS[dst]
'''


def converter_test(kind: str, cases: list[tuple], bad_unit: str) -> str:
    body = "\n".join(
        f"    assert_close(convert({v!r}, \"{s}\", \"{d}\"), {e!r})" for v, s, d, e in cases
    )
    return f'''
import pytest

from tests.helpers import assert_close
from unitconv.converters.{kind} import convert
from unitconv.errors import ConversionError


def test_{kind}_conversions():
{body}


def test_{kind}_unknown_unit():
    with pytest.raises(ConversionError):
        convert(1.0, "{bad_unit}", "{cases[0][2]}")
'''


def converters_init(kinds: list[str]) -> str:
    imports = "\n".join(f"from unitconv.converters import {k}  # noqa: F401" for k in sorted(kinds))
    return f'''
"""Importing a converter module registers it. Keep the imports sorted."""

{imports}
'''


def readme(rows: list[str]) -> str:
    table = "\n".join(rows)
    return f'''
# unitconv

Tiny unit conversion CLI.

```
python -m unitconv.cli length 5 km mi
```

## Supported conversions

| kind | units |
|------|-------|
{table}
'''


def changelog(entries: list[str]) -> str:
    lines = "\n".join(entries)
    return f'''
# Changelog

## Unreleased

{lines}

## 0.1.0

- Initial release with the `length` converter.
'''


BASE_FILES = {
    ".gitignore": "__pycache__/\n*.pyc\n.pytest_cache/\n",
    "unitconv/__init__.py": '__version__ = "0.1.0"\n',
    "unitconv/errors.py": '''
class ConversionError(ValueError):
    """Raised for any invalid conversion request."""


def unknown_unit(kind: str, unit: str, factors: dict) -> ConversionError:
    known = ", ".join(sorted(factors))
    return ConversionError(f"unknown {kind} unit {unit!r} (known: {known})")
''',
    "unitconv/registry.py": '''
"""Converter registry. Converters register themselves with @register("<kind>")."""

from typing import Callable

CONVERTERS: dict[str, Callable[[float, str, str], float]] = {}


def register(kind: str):
    def decorator(fn):
        if kind in CONVERTERS:
            raise RuntimeError(f"converter {kind!r} registered twice")
        CONVERTERS[kind] = fn
        return fn
    return decorator


def get_converter(kind: str):
    from unitconv.errors import ConversionError
    import unitconv.converters  # noqa: F401  (registers all converters)

    if kind not in CONVERTERS:
        raise ConversionError(f"unknown kind {kind!r} (known: {', '.join(sorted(CONVERTERS))})")
    return CONVERTERS[kind]
''',
    "unitconv/cli.py": '''
import argparse
import sys

from unitconv.errors import ConversionError
from unitconv.registry import get_converter


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="unitconv")
    parser.add_argument("kind")
    parser.add_argument("value", type=float)
    parser.add_argument("src")
    parser.add_argument("dst")
    args = parser.parse_args(argv)
    try:
        result = get_converter(args.kind)(args.value, args.src, args.dst)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"{result:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
    "unitconv/converters/__init__.py": converters_init(["length"]),
    "unitconv/converters/length.py": converter_module(
        "length", "meter", {"m": 1.0, "km": 1000.0, "cm": 0.01, "mi": 1609.344, "ft": 0.3048},
        "Length conversions."),
    "tests/__init__.py": "",
    "tests/helpers.py": '''
def assert_close(actual: float, expected: float, rel: float = 1e-6):
    assert abs(actual - expected) <= rel * max(1.0, abs(expected)), f"{actual} != {expected}"
''',
    "tests/test_length.py": converter_test("length", [(5.0, "km", "m", 5000.0), (1.0, "mi", "ft", 5280.0)], "parsec"),
    "README.md": readme(["| length | m, km, cm, mi, ft |"]),
    "CHANGELOG.md": changelog([]),
}

COMMITS = [
    {
        "split": "learning",
        "title": "Add weight converter",
        "files": {
            "unitconv/converters/__init__.py": converters_init(["length", "weight"]),
            "unitconv/converters/weight.py": converter_module(
                "weight", "kilogram", {"kg": 1.0, "g": 0.001, "lb": 0.45359237, "oz": 0.028349523125},
                "Weight conversions."),
            "tests/test_weight.py": converter_test("weight", [(2.0, "kg", "g", 2000.0), (1.0, "lb", "oz", 16.0)], "stone"),
            "README.md": readme(["| length | m, km, cm, mi, ft |", "| weight | kg, g, lb, oz |"]),
            "CHANGELOG.md": changelog(["- Added `weight` converter (kg, g, lb, oz)."]),
        },
        "query": (
            "Users keep asking for weight conversions. Please add support for converting between "
            "kilograms, grams, pounds and ounces, so that e.g. `python -m unitconv.cli weight 2 kg lb` "
            "works. Make sure it is covered by tests and documented."
        ),
    },
    {
        "split": "learning",
        "title": "Add volume converter",
        "files": {
            "unitconv/converters/__init__.py": converters_init(["length", "volume", "weight"]),
            "unitconv/converters/volume.py": converter_module(
                "volume", "liter", {"l": 1.0, "ml": 0.001, "gal": 3.785411784, "cup": 0.2365882365},
                "Volume conversions."),
            "tests/test_volume.py": converter_test("volume", [(1.5, "l", "ml", 1500.0), (1.0, "gal", "cup", 16.0)], "barrel"),
            "README.md": readme(["| length | m, km, cm, mi, ft |", "| volume | l, ml, gal, cup |",
                                 "| weight | kg, g, lb, oz |"]),
            "CHANGELOG.md": changelog(["- Added `weight` converter (kg, g, lb, oz).",
                                       "- Added `volume` converter (l, ml, gal, cup)."]),
        },
        "query": (
            "Add volume conversions to unitconv: liters, milliliters, US gallons and US cups "
            "(e.g. `python -m unitconv.cli volume 1 gal l`). Include tests and keep the docs up to date."
        ),
    },
    {
        "split": "test",
        "title": "Add speed converter",
        "files": {
            "unitconv/converters/__init__.py": converters_init(["length", "speed", "volume", "weight"]),
            "unitconv/converters/speed.py": converter_module(
                "speed", "meter per second", {"m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704, "kn": 1852 / 3600},
                "Speed conversions."),
            "tests/test_speed.py": converter_test("speed", [(36.0, "km/h", "m/s", 10.0), (1.0, "kn", "km/h", 1.852)], "mach"),
            "README.md": readme(["| length | m, km, cm, mi, ft |", "| speed | m/s, km/h, mph, kn |",
                                 "| volume | l, ml, gal, cup |", "| weight | kg, g, lb, oz |"]),
            "CHANGELOG.md": changelog(["- Added `weight` converter (kg, g, lb, oz).",
                                       "- Added `volume` converter (l, ml, gal, cup).",
                                       "- Added `speed` converter (m/s, km/h, mph, kn)."]),
        },
        "query": (
            "We need speed conversions: meters per second, kilometers per hour, miles per hour and "
            "knots (e.g. `python -m unitconv.cli speed 100 km/h mph`)."
        ),
    },
]


def main():
    argparse.ArgumentParser(description="Build the toy benchmark into examples/toy/build/").parse_args()
    if BUILD.exists():
        shutil.rmtree(BUILD)
    REPO.mkdir(parents=True)
    git("init", "-q")
    write(BASE_FILES)
    git("add", "-A")
    git("commit", "-q", "-m", "Initial unitconv with length converter")

    (BUILD / "snapshots").mkdir()
    (BUILD / "diffs").mkdir()
    records = []
    for commit in COMMITS:
        write(commit["files"])
        git("add", "-A")
        git("commit", "-q", "-m", commit["title"])
        sha = git("rev-parse", "HEAD").strip()
        sha10 = sha[:10]
        snapshot = BUILD / "snapshots" / f"snapshot_{sha10}.tar.gz"
        snapshot.write_bytes(subprocess.run(
            ["git", "archive", "--format=tar.gz", f"{sha}~1"], cwd=REPO, capture_output=True, check=True,
        ).stdout)
        diff = BUILD / "diffs" / f"commit_{sha10}.diff"
        diff.write_text(git("diff", f"{sha}~1", sha, "--no-color"), encoding="utf-8")
        records.append({**commit, "sha": sha, "sha10": sha10})

    learning = [
        {
            "sha": r["sha"],
            "repo": "unitconv",
            "title": r["title"],
            "synthetic_query": r["query"],
            "diff_path": f"diffs/commit_{r['sha10']}.diff",
            "snapshot_path": f"snapshots/snapshot_{r['sha10']}.tar.gz",
        }
        for r in records if r["split"] == "learning"
    ]
    tasks = [
        {
            "id": f"commit_{r['sha10']}",
            "repo": "unitconv",
            "synthetic_query": r["query"],
            "oracle_diff": f"diffs/commit_{r['sha10']}.diff",
            "snapshot_path": f"snapshots/snapshot_{r['sha10']}.tar.gz",
            "learning_commits": learning,
        }
        for r in records if r["split"] == "test"
    ]
    with open(BUILD / "benchmark_tasks.jsonl", "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")
    print(f"Toy benchmark written to {BUILD}: {len(learning)} learning commits, {len(tasks)} test task(s)")
    print(f"  tasks file: {BUILD / 'benchmark_tasks.jsonl'}")


if __name__ == "__main__":
    main()
