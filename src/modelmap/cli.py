"""The ``modelmap`` command.

``modelmap scan``       scan and print a summary, or the inventory as JSON
``modelmap list``       scan, or read an inventory, and print every model
``modelmap map``        scan, or read an inventory, and write the treemap
``modelmap providers``  show where each store is looked for on this machine
``modelmap rm``         remove models, after showing what goes and asking
``modelmap serve``      serve the map on this machine and open it in a browser
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, TextIO

from modelmap import __version__
from modelmap.providers import PROVIDERS
from modelmap.providers.base import Context
from modelmap.records import ModelRecord, Scan
from modelmap.removal import TRASH_NOTE, Plan, execute, find, plan
from modelmap.render_html import render_html
from modelmap.render_svg import DISPLAY_NAMES, render_svg
from modelmap.scanner import Options, redact, scan
from modelmap.serve import App, make_server
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


def _port_arg(text: str) -> int:
    try:
        port = int(text)
    except ValueError:
        port = -1
    if not 0 <= port <= 65535:
        raise argparse.ArgumentTypeError(f"not a port: {text}")
    return port


def _parser() -> argparse.ArgumentParser:
    what = argparse.ArgumentParser(add_help=False)
    what.add_argument("paths", nargs="*", metavar="PATH", help="files or directories to scan")
    what.add_argument(
        "--only", metavar="NAMES", help="comma-separated providers to run, for example hf,ollama"
    )
    what.add_argument(
        "--no-defaults", action="store_true", help="scan only the given paths, not the known stores"
    )
    sweeping = what.add_mutually_exclusive_group()
    sweeping.add_argument(
        "--no-sweep", action="store_true", help="do not sweep the home directory for loose models"
    )
    sweeping.add_argument(
        "--sweep",
        action="store_true",
        help="sweep the home directory too, even when PATHs or --only are given",
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

    map_cmd = commands.add_parser("map", parents=[what], help="write the treemap")
    map_cmd.add_argument(
        "--from-json", metavar="FILE", help="read an inventory instead of scanning"
    )
    map_cmd.add_argument(
        "-o", "--output", metavar="FILE", help="file to write (default modelmap.html or .svg)"
    )
    map_cmd.add_argument(
        "--svg", action="store_true", help="write a static picture, not the interactive page"
    )
    map_cmd.add_argument(
        "--disk", action="store_true", help="size tiles by space allocated on disk, not file size"
    )
    map_cmd.add_argument("--open", action="store_true", help="open the result in a browser")

    serve_cmd = commands.add_parser(
        "serve",
        parents=[what],
        help="serve the map on this machine and open it in a browser",
        description=(
            "Scan, then serve the map from 127.0.0.1 until Ctrl-C. The address it prints "
            "holds a token for this run; the server refuses requests without it."
        ),
    )
    serve_cmd.add_argument(
        "--from-json", metavar="FILE", help="read an inventory instead of scanning"
    )
    serve_cmd.add_argument(
        "--disk", action="store_true", help="size tiles by space allocated on disk, not file size"
    )
    serve_cmd.add_argument(
        "--port", type=_port_arg, default=0, help="port on 127.0.0.1 (default: any free port)"
    )
    serve_cmd.add_argument(
        "--no-open", action="store_true", help="print the address; do not open a browser"
    )
    serve_cmd.add_argument(
        "--read-only", action="store_true", help="offer no Remove button; --from-json implies it"
    )

    commands.add_parser("providers", help="show where each store is looked for")

    rm_cmd = commands.add_parser(
        "rm",
        help="remove models, after showing what goes and asking",
        description=(
            "Remove models. Each TARGET is provider:name as `modelmap list` shows it, an "
            "Ollama tag, or a path. Ollama models and Hugging Face cache repositories are "
            "removed with the tool's own command; anything else moves to the Trash."
        ),
    )
    rm_cmd.add_argument("targets", nargs="+", metavar="TARGET", help="what to remove")
    rm_cmd.add_argument("--dry-run", action="store_true", help="show what would go; remove nothing")
    rm_cmd.add_argument("-y", "--yes", action="store_true", help="do not ask before removing")
    return parser


def _scan(args: argparse.Namespace, ctx: Context) -> Scan:
    options = Options(
        paths=args.paths,
        only=args.only.split(",") if args.only else None,
        defaults=not args.no_defaults,
        sweep=False if args.no_sweep else True if args.sweep else None,
        excludes=args.exclude,
        min_size=args.min_size,
        max_depth=args.max_depth,
    )
    result = scan(ctx, options)
    return redact(result, ctx.home) if args.redact else result


def _write_map(result: Scan, args: argparse.Namespace, out: TextIO) -> None:
    text = (
        render_svg(result, use_disk=args.disk)
        if args.svg
        else render_html(result, use_disk=args.disk)
    )
    path = os.path.abspath(args.output or ("modelmap.svg" if args.svg else "modelmap.html"))
    # Names that are not valid UTF-8 were replaced when the text was built.
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Wrote {_text(path)}", file=out)
    if args.open:
        webbrowser.open(Path(path).as_uri())


def _serve(args: argparse.Namespace, ctx: Context, out: TextIO, result: Scan) -> int:
    """The ``serve`` command: runs until Ctrl-C."""
    if args.from_json:
        inventory = args.from_json

        def rescan() -> Scan:
            with open(inventory, encoding="utf-8") as fh:
                return Scan.from_json(fh.read())

    else:

        def rescan() -> Scan:
            return _scan(args, ctx)

    def log(line: str) -> None:
        print(_text(line), file=out, flush=True)

    try:
        app = App(
            result,
            rescan,
            use_disk=args.disk,
            env=ctx.env,
            home=str(ctx.home),
            # An inventory describes another machine's disk, or this one's at
            # another time: nothing in it is removed.
            read_only=args.read_only or bool(args.from_json),
        )
        server = make_server(app, args.port, log)
    except OSError as exc:
        print(f"Cannot listen on 127.0.0.1 port {args.port}: {exc.strerror or exc}", file=out)
        return 1
    print(f"Serving the map at {server.url}", file=out)
    print("Press Ctrl-C to stop.", file=out, flush=True)
    if not args.no_open:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopped.", file=out)
    finally:
        server.server_close()
    return 0


def _describe_plan(step: Plan, out: TextIO) -> None:
    record = step.record
    store = DISPLAY_NAMES.get(record.provider, record.provider)
    print(f"  {_text(record.name)} ({store}, {format_size(record.size_bytes)})", file=out)
    if step.action == "trash":
        for path in step.paths:  # all of them: this is what the question asks about
            print(f"    {_text(path)}", file=out)
    else:
        print(f"    {_text(record.path)}", file=out)
    print(f"    {_text(step.note)}", file=out)


def _remove(
    args: argparse.Namespace, ctx: Context, out: TextIO, ask: Callable[[str], str] | None
) -> int:
    """The ``rm`` command. Nothing is removed that was not listed first."""
    paths = []
    for target in args.targets:
        found = os.path.expanduser(target)
        if os.path.lexists(found):
            # A hub repository is read with its cache, which names it.
            prefix = os.path.basename(found.rstrip(os.sep)).partition("--")[0]
            hub_repo = prefix in ("models", "datasets", "spaces") and "--" in found
            paths.append(os.path.dirname(found.rstrip(os.sep)) if hub_repo else found)
    result = scan(ctx, Options(paths=paths, sweep=True))
    chosen: list[ModelRecord] = []
    for target in args.targets:
        matches = find(result.records, target)
        if not matches:
            print(f"No model matches {_text(target)}. `modelmap list` shows the names.", file=out)
            return 1
        if len(matches) > 1:
            print(
                f"{_text(target)} matches {len(matches)} entries; name one by its path:", file=out
            )
            for match in matches:
                print(f"  {_text(match.path)}", file=out)
            return 1
        if matches[0] not in chosen:
            chosen.append(matches[0])

    live = [r.resolved for r in result.roots if r.provider == "ollama" and r.source != "layout"]
    steps = [plan(record, live, env=ctx.env, home=str(ctx.home)) for record in chosen]
    if args.yes:
        # A tag with settings of its own goes only if it was asked for by name.
        typed = set(args.targets)
        for n, step in enumerate(steps):
            unnamed = [tag for tag in step.variants if tag not in typed]
            if unnamed:
                steps[n] = Plan(
                    step.record,
                    "refuse",
                    f"{step.note} With --yes, name each of these tags to remove it: "
                    f"{', '.join(unnamed)}.",
                )
    refused = [step for step in steps if step.action == "refuse"]
    total = sum(step.record.size_bytes for step in steps)
    heading = "Would remove" if args.dry_run else "Will remove"
    count = len(steps)
    print(
        f"{heading} {count} {'entry' if count == 1 else 'entries'}, {format_size(total)}:", file=out
    )
    for step in steps:
        print(file=out)
        _describe_plan(step, out)
    if refused:
        print(f"\nNothing was removed: {len(refused)} of these cannot be removed.", file=out)
        return 1
    if args.dry_run:
        print("\nNothing was removed (--dry-run).", file=out)
        return 0
    if not args.yes:
        if ask is None:
            if not sys.stdin.isatty():
                print("\nNothing was removed: add --yes to remove without asking.", file=out)
                return 1
            ask = input
        print(file=out)
        out.flush()
        if ask("Remove? [y/N] ").strip().lower() not in ("y", "yes"):
            print("Nothing was removed.", file=out)
            return 1

    failed = 0
    for step in steps:
        done, problem = execute(step, ctx.env)
        name = _text(step.record.name)
        if done:
            print(f"Removed {name}.", file=out)
        else:
            failed += 1
            print(f"Could not remove {name}: {_text(problem)}", file=out)
    if any(step.action == "trash" for step in steps) and failed < count:
        print(TRASH_NOTE, file=out)
    return 1 if failed else 0


def main(
    argv: Sequence[str] | None = None,
    ctx: Context | None = None,
    out: TextIO | None = None,
    ask: Callable[[str], str] | None = None,
) -> int:
    """Run the command line.

    :param argv: Arguments; ``sys.argv[1:]`` by default.
    :param ctx: The machine to scan; the running one by default.
    :param out: Where to print; standard output by default.
    :param ask: Asks the person a question and returns the answer; the
        terminal by default, and only when standard input is one.
    :return: Exit status.
    """
    parser = _parser()
    args = parser.parse_args(argv)
    ctx = ctx or Context.current()
    out = out or sys.stdout
    if args.command == "providers":
        _print_providers(ctx, out)
        return 0
    if args.command == "rm":
        return _remove(args, ctx, out, ask)
    if args.no_defaults and not args.paths and not getattr(args, "from_json", None):
        parser.error("--no-defaults needs at least one PATH to scan")
    if args.no_defaults and args.sweep:
        # The sweep is part of the default scan; the scanner runs it only then.
        parser.error("--sweep cannot go with --no-defaults")
    if args.command == "serve" and args.redact:
        parser.error("--redact is for sharing a map; serve shows this machine to its owner")
    if args.command in ("list", "map", "serve") and args.from_json:
        # An inventory is read as it is. It does not say what the home
        # directory was, so it cannot be redacted here, only when it is made.
        scanning = (
            args.paths or args.only or args.no_sweep or args.sweep or args.exclude or args.redact
        )
        if scanning or args.max_depth is not None:
            parser.error("--from-json reads an inventory as it is: give no PATH or scan options")
        try:
            with open(args.from_json, encoding="utf-8") as fh:
                result = Scan.from_json(fh.read())
        except (OSError, ValueError) as exc:
            parser.error(f"cannot read inventory {args.from_json}: {exc}")
    else:
        result = _scan(args, ctx)

    if args.command == "list":
        _print_list(result, args, out)
    elif args.command == "map":
        _write_map(result, args, out)
    elif args.command == "serve":
        return _serve(args, ctx, out, result)
    elif args.json:
        print(result.to_json(), file=out)
    else:
        _print_summary(result, out)
    unreadable = result.warnings or any(r.issues for r in result.records)
    return 1 if args.strict and unreadable else 0


if __name__ == "__main__":
    sys.exit(main())
