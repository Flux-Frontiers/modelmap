"""The ``modelmap`` command.

``modelmap scan``       scan and print a summary, or the inventory as JSON
``modelmap list``       scan, or read an inventory, and print every model
``modelmap providers``  show where each store is looked for on this machine
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TextIO

from modelmap import __version__
from modelmap.providers import PROVIDERS
from modelmap.providers.base import Context
from modelmap.records import ModelRecord, Scan
from modelmap.scanner import Options, redact, scan
from modelmap.units import format_count, format_size, parse_size

_SORTS: dict[str, Callable[[ModelRecord], Any]] = {
    "size": lambda r: (-r.size_bytes, r.name),
    "name": lambda r: r.name.lower(),
    "params": lambda r: (-(r.params or 0), r.name),
    "mtime": lambda r: (-(r.mtime or 0), r.name),
}
_GROUPS = ("provider", "format", "family", "kind", "none")


def _text(value: object) -> str:
    """Text safe to print: a name that is not valid UTF-8 is escaped."""
    return str(value).encode("utf-8", "backslashreplace").decode("utf-8")


def _table(rows: Iterable[Sequence[str]], right: Sequence[int] = ()) -> list[str]:
    """Align rows into columns; ``right`` lists the right-aligned columns."""
    rows = [[_text(cell) for cell in row] for row in rows]
    if not rows:
        return []
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for row in rows:
        cells = [
            cell.rjust(width) if i in right else cell.ljust(width)
            for i, (cell, width) in enumerate(zip(row, widths))
        ]
        lines.append("  ".join(cells).rstrip())
    return lines


def _size(record: ModelRecord) -> str:
    """Size, or who was charged when the record shares all its files."""
    if record.size_bytes == 0 and record.shared_with:
        return "shared"
    return format_size(record.size_bytes)


def _notes(record: ModelRecord) -> str:
    notes = []
    if record.aliases:
        notes.append("also " + ", ".join(record.aliases))
    if record.shared_with:
        notes.append("shared with " + ", ".join(record.shared_with))
    if record.kind != "model":
        notes.append(record.kind)
    notes += [f"! {issue}" for issue in record.issues]
    return "; ".join(notes)


def _print_list(scan_result: Scan, args: argparse.Namespace, out: TextIO) -> None:
    records = [r for r in scan_result.records if args.all or r.kind in ("model", "auxiliary")]
    records.sort(key=_SORTS[args.sort])
    groups: dict[str, list[ModelRecord]] = {}
    for record in records:
        key = "" if args.group == "none" else str(getattr(record, args.group) or "unknown")
        groups.setdefault(key, []).append(record)
    header = ["SIZE", "PARAMS", "QUANT", "FORMAT", "PROVIDER", "NAME", "NOTES"]
    for key in sorted(groups, key=lambda k: -sum(r.size_bytes for r in groups[k])):
        members = groups[key]
        if key:
            total = format_size(sum(r.size_bytes for r in members))
            print(f"\n{_text(key)}  {total}  {len(members)} records", file=out)
        rows = [header] + [
            [
                _size(r),
                format_count(r.params, r.params_exact),
                r.quant or "",
                r.format,
                r.provider,
                r.name,
                _notes(r),
            ]
            for r in members
        ]
        print("\n".join(_table(rows, right=(0, 1))), file=out)
    hidden = len(scan_result.records) - len(records)
    if hidden:
        print(f"\n{hidden} datasets, caches and other entries not shown; use --all", file=out)


def _print_summary(scan_result: Scan, out: TextIO) -> None:
    by_provider: dict[str, list[ModelRecord]] = {}
    for record in scan_result.records:
        by_provider.setdefault(record.provider, []).append(record)
    rows = [["PROVIDER", "SIZE", "ON DISK", "MODELS", "OTHER", "ISSUES"]]
    for provider, records in sorted(
        by_provider.items(), key=lambda kv: -sum(r.size_bytes for r in kv[1])
    ):
        models = sum(1 for r in records if r.kind in ("model", "auxiliary"))
        rows.append(
            [
                provider,
                format_size(sum(r.size_bytes for r in records)),
                format_size(sum(r.disk_bytes for r in records)),
                str(models),
                str(len(records) - models),
                str(sum(1 for r in records if r.issues)),
            ]
        )
    rows.append(
        [
            "total",
            format_size(scan_result.size_bytes),
            format_size(scan_result.disk_bytes),
            str(sum(1 for r in scan_result.records if r.kind in ("model", "auxiliary"))),
            "",
            "",
        ]
    )
    print("\n".join(_table(rows, right=(1, 2, 3, 4, 5))), file=out)
    roots = sum(1 for r in scan_result.roots if r.exists)
    print(f"\n{roots} locations scanned in {scan_result.elapsed_s:.1f} s", file=out)
    for warning in scan_result.warnings:
        print(_text(f"warning: {warning.path}: {warning.message}".replace(": :", ":")), file=out)


def _print_providers(ctx: Context, out: TextIO) -> None:
    rows = [["PROVIDER", "FOUND", "FROM", "PATH"]]
    for provider in PROVIDERS:
        for root in provider.roots(ctx):
            path = root.path if root.resolved == root.path else f"{root.path} -> {root.resolved}"
            rows.append([provider.NAME, "yes" if root.exists else "no", root.source, path])
    print("\n".join(_table(rows)), file=out)


def _size_arg(text: str) -> int:
    try:
        return parse_size(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _parser() -> argparse.ArgumentParser:
    what = argparse.ArgumentParser(add_help=False)
    what.add_argument("paths", nargs="*", metavar="PATH", help="files or directories to scan")
    what.add_argument(
        "--only", metavar="NAMES", help="comma-separated providers to run, for example hf,ollama"
    )
    what.add_argument(
        "--no-defaults", action="store_true", help="scan only the given paths, not the known stores"
    )
    what.add_argument(
        "--no-sweep", action="store_true", help="do not sweep the home directory for loose models"
    )
    what.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="GLOB",
        help="skip matching files and directories",
    )
    what.add_argument(
        "--min-size",
        type=_size_arg,
        default=Options.min_size,
        metavar="SIZE",
        help="smallest checkpoint to list when only its extension says it is a model (default 50MB)",
    )
    what.add_argument(
        "--max-depth", type=int, metavar="N", help="levels to descend below each given path"
    )
    what.add_argument(
        "--redact",
        action="store_true",
        help="replace the home directory with ~ and drop the host name",
    )
    what.add_argument(
        "--strict", action="store_true", help="exit with status 1 if anything could not be read"
    )

    parser = argparse.ArgumentParser(
        prog="modelmap", description="Find local LLM model files and report them by size."
    )
    parser.add_argument("--version", action="version", version=f"modelmap {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    scan_cmd = commands.add_parser("scan", parents=[what], help="scan and print a summary")
    scan_cmd.add_argument("--json", action="store_true", help="print the inventory as JSON")

    list_cmd = commands.add_parser("list", parents=[what], help="print every model found")
    list_cmd.add_argument(
        "--from-json", metavar="FILE", help="read an inventory instead of scanning"
    )
    list_cmd.add_argument("--sort", choices=sorted(_SORTS), default="size")
    list_cmd.add_argument("--group", choices=_GROUPS, default="provider")
    list_cmd.add_argument(
        "--all", action="store_true", help="include datasets, caches and other entries"
    )

    commands.add_parser("providers", help="show where each store is looked for")
    return parser


def _scan(args: argparse.Namespace, ctx: Context) -> Scan:
    options = Options(
        paths=args.paths,
        only=args.only.split(",") if args.only else None,
        defaults=not args.no_defaults,
        sweep=False if args.no_sweep else None,
        excludes=args.exclude,
        min_size=args.min_size,
        max_depth=args.max_depth,
    )
    result = scan(ctx, options)
    return redact(result, ctx.home) if args.redact else result


def main(
    argv: Sequence[str] | None = None, ctx: Context | None = None, out: TextIO | None = None
) -> int:
    """Run the command line.

    :param argv: Arguments; ``sys.argv[1:]`` by default.
    :param ctx: The machine to scan; the running one by default.
    :param out: Where to print; standard output by default.
    :return: Exit status.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    ctx = ctx or Context.current()
    out = out or sys.stdout
    if args.command == "providers":
        _print_providers(ctx, out)
        return 0
    if args.no_defaults and not args.paths and not getattr(args, "from_json", None):
        parser.error("--no-defaults needs at least one PATH to scan")
    if args.command == "list" and args.from_json:
        try:
            with open(args.from_json, encoding="utf-8") as fh:
                result = Scan.from_json(fh.read())
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read inventory {args.from_json}: {exc}")
    else:
        result = _scan(args, ctx)

    if args.command == "list":
        _print_list(result, args, out)
    elif args.json:
        print(result.to_json(), file=out)
    else:
        _print_summary(result, out)
    unreadable = result.warnings or any(r.issues for r in result.records)
    return 1 if args.strict and unreadable else 0


if __name__ == "__main__":
    sys.exit(main())
