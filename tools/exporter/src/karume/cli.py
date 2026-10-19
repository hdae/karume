"""The `karume` command — subcommand dispatch for dist / migrate / verify.

    karume dist --pipeline siglip2   # fails on core alone, whose accepted set is empty (NOTE below)
    karume verify ../../models/karume-irodori-v4.1-small/v4.1-small/dit/model.i8.krm
    # <old-mirror> = a mirror of the old karume/4 form (safetensors), outside the repository
    karume migrate <old-mirror>/<series>/model.f16.safetensors --out /tmp/krm \
        --license apache-2.0 --graph-name vision

MUST: the CLI **does not interpret arguments**. It dispatches on the first word and passes the rest
unchanged to the matching `main(argv)` (`--help` passes through too, so usage comes from each
command's own parser). A copy of the arguments here would let the exclusivity rules the command
owns (**the silent-wrong-value checks**) go missing in the CLI's copy, and nothing would fail when
only one side goes stale.

NOTE: the `export-*` subcommands **have all left this roster** — every export script moved to
`tools/export-recipes/<family>/` and is neither at the repository root nor in the wheel (ADR 0065
stages 3 and 4). Run them with `uv run python -m <family>.export` (from the export-recipes root).
The mechanism that loaded a script by path (formerly `load_script`) lost its users and was removed
too — leaving half of it would keep a foothold for the backflow of "the wheel's CLI reading what is
not in the wheel".

NOTE: `karume dist` runs on core's accepted set ({@link karume.dist.PIPELINES} — **empty** today),
so building a family uses the repo driver (`tools/export-recipes/dist.py`). The command stays
because the assembly engine itself is core's responsibility (ADR 0065 decision 1).

MUST: dispatch imports lazily — keep `karume verify` startable without importing `karume.dist` (and
the other way round).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence


def run_dist(argv: Sequence[str]) -> None:
    from karume import dist

    return dist.main(argv)


def run_migrate(argv: Sequence[str]) -> None:
    from karume import migrate

    return migrate.main(argv)


def run_verify(argv: Sequence[str]) -> None:
    from karume import verify

    return verify.main(argv)


#: Subcommand name → (handler, the one line shown in the listing). The order is the `--help` order.
COMMANDS: Mapping[str, tuple[Callable[[Sequence[str]], None], str]] = {
    "dist": (run_dist, "assemble a distribution directory and write karume.json / README.md"),
    "migrate": (
        run_migrate,
        "move an old distribution to containers (krm / krg); input is read-only",
    ),
    "verify": (run_verify, "verify a container (krm / krg) against every rule"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="karume",
        description="Karume exporter — lowers a torch.export-ed model to IR and a krm container",
    )
    subcommands = parser.add_subparsers(
        dest="command", required=True, metavar=f"{{{','.join(COMMANDS)}}}"
    )
    for name, (_, summary) in COMMANDS.items():
        # add_help=False: so that `karume <cmd> --help` passes through to the command's own parser.
        subcommands.add_parser(name, help=summary, add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args, rest = build_parser().parse_known_args(argv)
    handler, _ = COMMANDS[args.command]
    handler(rest)


if __name__ == "__main__":
    main()
