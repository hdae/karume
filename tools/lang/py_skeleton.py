"""Code skeleton of a Python file: the AST without its docstrings (comments mode), or without its
docstrings and string constants (strings mode). The Python counterpart of `code-skeleton.ts` for
the language migration (ADR 0130, step 2-9f).

A translation that only rewrites comments and docstrings keeps the comments-mode skeleton. A
translation that also rewrites messages keeps the strings-mode skeleton; the strings mode lists the
constants that changed, and a changed constant that held no Japanese before is a code change (the
translation touched something it was not meant to touch).

Formatting is invisible to both modes: the skeleton is the AST, so a `ruff format` wrap does not
show up as a change.

Usage (from the repository root; `python` = `uv run --project tools/exporter python`):
  python tools/lang/py_skeleton.py --compare <git-ref> [--head <git-ref>] [file...]
      compares the comments-mode skeleton of the changed .py files between the ref and the working
      tree (or `--head`); exit 1 when a skeleton differs
  python tools/lang/py_skeleton.py --strings [--json] --compare <ref> [--head <ref>] [file...]
      the same with the strings skeleton; lists every changed constant (`--json`: as one JSON
      array of {file, index, before, after, japaneseBefore} | {file, codeDiffers: true});
      exit 1 when a skeleton differs or a constant without Japanese changed
  python tools/lang/py_skeleton.py [--strings] <file>
      prints the skeleton of one file
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Same character classes as `scan-japanese.ts` (the scanner this migration is measured with).
JAPANESE = re.compile(r"[぀-ゟ゠-ヿ㐀-䶿一-鿿々、。「」・〈〉]")

# The strings-mode cell marker; a string constant in the skeleton is replaced by it.
LITERAL_MARK = "\x01"
# Inside a lifted f-string, an interpolation is shown as this token (the expression stays code).
INTERPOLATION = "${}"

Mode = Literal["comments", "strings"]


@dataclass(frozen=True)
class Skeleton:
    text: str
    literals: tuple[str, ...]


@dataclass(frozen=True)
class LiteralChange:
    index: int
    before: str
    after: str
    japanese_before: bool


def _is_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


class _Lift(ast.NodeTransformer):
    """Drops docstrings; in strings mode also replaces every string constant by the marker and
    collects the constants in visiting order (the order of the source)."""

    def __init__(self, mode: Mode) -> None:
        self.mode = mode
        self.literals: list[str] = []

    def visit(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = node.body
            if body and _is_docstring(body[0]):
                node.body = body[1:] if len(body) > 1 else [ast.Pass()]
        return super().visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.mode == "strings" and isinstance(node.value, str):
            self.literals.append(node.value)
            return ast.Constant(value=LITERAL_MARK)
        return node

    def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
        if self.mode != "strings":
            return self.generic_visit(node)
        # One cell per f-string: its text parts joined by the interpolation token. The interpolated
        # expressions stay code (visited); their format specs stay code too (not lifted).
        parts: list[str] = []
        values: list[ast.expr] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                parts.append(INTERPOLATION)
                value.value = self.visit(value.value)
                values.append(value)
            else:
                values.append(self.visit(value))
        self.literals.append("".join(parts))
        node.values = [ast.Constant(value=LITERAL_MARK), *values]
        return node


def analyze(filename: str, source: str, mode: Mode = "comments") -> Skeleton:
    """The skeleton of `source`; raises SyntaxError (prefixed with the file name) when it does not
    parse."""
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as error:
        raise SyntaxError(f"cannot parse {filename}: {error}") from error
    lift = _Lift(mode)
    tree = lift.visit(tree)
    return Skeleton(text=ast.dump(tree), literals=tuple(lift.literals))


def skeleton_of(filename: str, source: str) -> str:
    return analyze(filename, source, "comments").text


def literal_changes(before: Skeleton, after: Skeleton) -> list[LiteralChange]:
    """The constants that differ between two strings-mode skeletons of the same code."""
    if before.text != after.text or len(before.literals) != len(after.literals):
        raise ValueError("the skeletons differ: the constants cannot be aligned")
    return [
        LiteralChange(
            index=index, before=a, after=b, japanese_before=JAPANESE.search(a) is not None
        )
        for index, (a, b) in enumerate(zip(before.literals, after.literals, strict=True))
        if a != b
    ]


def _git(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout if result.returncode == 0 else None


def _git_show(root: Path, ref: str, file: str) -> str | None:
    return _git(root, "show", f"{ref}:{file}")


def _changed_python_files(root: Path, base: str, head: str | None) -> list[str]:
    refs = [base] if head is None else [base, head]
    out = _git(root, "diff", "--name-only", "--diff-filter=ACMR", *refs, "--", "*.py") or ""
    return [line for line in out.splitlines() if line]


def _first_difference(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def _shown(text: str) -> str:
    return json.dumps(text[:80] + ("…" if len(text) > 80 else ""), ensure_ascii=False)


def main(argv: list[str]) -> int:
    root = Path.cwd()
    base: str | None = None
    head: str | None = None
    mode: Mode = "comments"
    as_json = False
    files: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--compare":
            base = argv[i + 1]
            i += 1
        elif arg == "--head":
            head = argv[i + 1]
            i += 1
        elif arg == "--strings":
            mode = "strings"
        elif arg == "--json":
            as_json = True
        else:
            files.append(arg)
        i += 1
    if base is None:
        if len(files) != 1:
            raise SystemExit(
                "usage: py_skeleton.py [--strings] --compare <ref> [--head <ref>] [file...]"
                " | <file>"
            )
        print(analyze(files[0], (root / files[0]).read_text(encoding="utf-8"), mode).text)
        return 0
    targets = files if files else _changed_python_files(root, base, head)
    differing = 0
    changed = 0
    without_japanese = 0
    report: list[dict[str, object]] = []
    for file in targets:
        before = _git_show(root, base, file)
        after = (
            (root / file).read_text(encoding="utf-8")
            if head is None
            else _git_show(root, head, file)
        )
        if before is None or after is None:
            differing += 1
            if as_json:
                report.append({"file": file, "codeDiffers": True})
            else:
                print(f"{file}: added or removed (not comparable)")
            continue
        a = analyze(file, before, mode)
        b = analyze(file, after, mode)
        at = _first_difference(a.text, b.text)
        if at >= 0:
            differing += 1
            if as_json:
                report.append({"file": file, "codeDiffers": True})
            else:
                print(f"{file}: code differs near: …{a.text[max(0, at - 40) : at + 60]}…")
            continue
        if mode != "strings":
            continue
        for change in literal_changes(a, b):
            changed += 1
            if not change.japanese_before:
                without_japanese += 1
            if as_json:
                report.append(
                    {
                        "file": file,
                        "index": change.index,
                        "before": change.before,
                        "after": change.after,
                        "japaneseBefore": change.japanese_before,
                    }
                )
                continue
            note = "" if change.japanese_before else " (no Japanese before — a code change)"
            print(
                f"{file}: literal {change.index}{note}: "
                f"{_shown(change.before)} -> {_shown(change.after)}"
            )
    if as_json:
        print(json.dumps(report, ensure_ascii=False))
    elif mode == "strings":
        print(
            f"compared {len(targets)} files against {base} (strings mode); {changed} literals "
            f"changed, {without_japanese} without Japanese before; {differing} differ beyond "
            "docstrings and strings"
        )
    else:
        print(f"compared {len(targets)} files against {base}; {differing} differ beyond docstrings")
    return 1 if differing > 0 or without_japanese > 0 else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
