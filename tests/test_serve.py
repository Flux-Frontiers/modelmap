"""``modelmap serve``: what it serves, and every request it must refuse."""

from __future__ import annotations

import base64
import hashlib
import http.client
import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from modelmap.cli import main
from modelmap.providers.base import Context
from modelmap.records import Scan
from modelmap.removal import plan
from modelmap.render_html import _SERVED_CSS, SCRIPTS, record_view, render_html
from modelmap.scanner import scan
from modelmap.serve import CSP, HOST, TOKEN_HEADER, App, Server, make_server, plan_json
from tests.fake_tools import Tools, install
from tests.fakemachine import GGML_F16, FakeMachine, write_gguf


class Running:
    """A served app on a free port, and a client that sets any header."""

    def __init__(self, server: Server, scans: list[Scan], log: list[str]) -> None:
        self.server = server
        self.app = server.app
        self.scans = scans
        self.log = log

    @property
    def host(self) -> str:
        return f"{HOST}:{self.server.port}"

    def request(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        body: bytes | None = None,
        host: str | None = None,
    ) -> tuple[int, dict[str, str], bytes]:
        conn = http.client.HTTPConnection(HOST, self.server.port, timeout=10)
        conn.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
        sent = {"Host": self.host if host is None else host, **(headers or {})}
        if body is not None:
            sent.setdefault("Content-Length", str(len(body)))
        for name, value in sent.items():
            conn.putheader(name, value)
        conn.endheaders(body)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, data

    def api(
        self, method: str, path: str, body: bytes | None = None, **headers: str
    ) -> tuple[int, dict[str, str], bytes]:
        sent = {TOKEN_HEADER: self.app.token, "Content-Type": "application/json"}
        sent.update({k.replace("_", "-"): v for k, v in headers.items()})
        return self.request(method, path, headers=sent, body=body)


@contextmanager
def start(ctx: Context, **options: Any) -> Iterator[Running]:
    """Serve a scan of ``ctx`` on a free port; every rescan is kept."""
    scans = [scan(ctx)]

    def rescan() -> Scan:
        scans.append(scan(ctx))
        return scans[-1]

    log: list[str] = []
    options.setdefault("env", ctx.env)
    options.setdefault("home", str(ctx.home))
    server = make_server(App(scans[0], rescan, **options), 0, log.append)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield Running(server, scans, log)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def served(ctx: Context) -> Iterator[Running]:
    with start(ctx) as running:
        yield running


def test_listens_on_loopback_only(served: Running) -> None:
    assert served.server.server_address[0] == "127.0.0.1"
    assert served.server.url == f"http://127.0.0.1:{served.server.port}/?t={served.app.token}"


def test_the_page_is_the_map_drawn_for_serving(served: Running) -> None:
    status, headers, body = served.request("GET", f"/?t={served.app.token}")
    assert status == 200
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert body.decode() == render_html(served.scans[0], served=True, removable=True)
    assert served.app.token not in body.decode()  # read from the address, not the page


@pytest.mark.parametrize("query", ["", "?t=", "?t=wrong", "?x=1"])
def test_the_page_needs_the_token(served: Running, query: str) -> None:
    status, _, body = served.request("GET", "/" + query)
    assert status == 403
    assert b"<svg" not in body


@pytest.mark.parametrize(
    "host", ["evil.example", "evil.example:{port}", "127.0.0.1:1", "127.0.0.1", "[::1]:{port}", ""]
)
def test_another_host_name_is_refused(served: Running, host: str) -> None:
    # DNS rebinding: a page whose name now points at 127.0.0.1 sends its own name.
    host = host.format(port=served.server.port)
    status, _, body = served.request("GET", f"/?t={served.app.token}", host=host)
    assert status == 403
    assert b"<svg" not in body
    assert served.api("GET", "/api/state")[0] == 200  # the same request, named right, passes


def test_localhost_is_accepted(served: Running) -> None:
    host = f"localhost:{served.server.port}"
    assert served.request("GET", f"/?t={served.app.token}", host=host)[0] == 200


@pytest.mark.parametrize("origin", ["https://evil.example", "null", "http://127.0.0.1:1"])
def test_another_origin_is_refused(served: Running, origin: str) -> None:
    before = len(served.scans)
    assert served.api("POST", "/api/rescan", b"{}", Origin=origin)[0] == 403
    assert served.api("GET", "/api/state", Origin=origin)[0] == 403
    assert len(served.scans) == before


def test_the_own_origin_is_accepted(served: Running) -> None:
    origin = f"http://{served.host}"
    assert served.api("POST", "/api/rescan", b"{}", Origin=origin)[0] == 200


@pytest.mark.parametrize("token", [None, "", "wrong"])
def test_the_api_needs_the_token(served: Running, token: str | None) -> None:
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers[TOKEN_HEADER] = token
    before = len(served.scans)
    assert served.request("GET", "/api/state", headers=headers)[0] == 403
    assert served.request("POST", "/api/rescan", headers=headers, body=b"{}")[0] == 403
    # A token in the address is for the page, not for the API.
    url = f"/api/rescan?t={served.app.token}"
    assert served.request("POST", url, headers=headers, body=b"{}")[0] == 403
    assert len(served.scans) == before


def test_state_reports_the_scan(served: Running) -> None:
    status, headers, body = served.api("GET", "/api/state")
    assert status == 200
    assert headers["content-type"] == "application/json"
    state = json.loads(body)
    first = served.scans[0]
    assert state["records"] == len(first.records)
    assert state["size_bytes"] == first.size_bytes
    assert state["disk_bytes"] == first.disk_bytes


def test_rescan_scans_again_and_redraws(served: Running, machine: FakeMachine) -> None:
    extra = machine.home / "extra" / "new-model.gguf"
    extra.parent.mkdir()
    write_gguf(
        extra,
        tensors=(("w", (1000, 1000), GGML_F16),),
        metadata={"general.architecture": "llama", "general.file_type": 1},
    )
    status, _, body = served.api("POST", "/api/rescan", b"{}")
    assert status == 200
    assert len(served.scans) == 2
    assert len(served.scans[1].records) == len(served.scans[0].records) + 1
    assert json.loads(body)["records"] == len(served.scans[1].records)
    page = served.request("GET", f"/?t={served.app.token}")[2].decode()
    assert page == render_html(served.scans[1], served=True, removable=True)


@pytest.mark.parametrize(
    ("headers", "body", "status"),
    [
        ({"Content-Type": "text/plain"}, b"{}", 415),
        ({"Content-Type": "application/x-www-form-urlencoded"}, b"a=1", 415),
        ({}, b"[]", 400),
        ({}, b"not json", 400),
        ({}, b"{" + b" " * 5000 + b"}", 413),
        ({"Content-Length": "x"}, b"{}", 411),
    ],
)
def test_bad_bodies_are_refused(
    served: Running, headers: dict[str, str], body: bytes, status: int
) -> None:
    before = len(served.scans)
    sent = {TOKEN_HEADER: served.app.token, "Content-Type": "application/json", **headers}
    assert served.request("POST", "/api/rescan", headers=sent, body=body)[0] == status
    assert len(served.scans) == before


def test_one_scan_at_a_time(served: Running) -> None:
    with served.app.lock:
        status, _, body = served.api("POST", "/api/rescan", b"{}")
    assert status == 409
    assert "Busy" in json.loads(body)["error"]
    assert len(served.scans) == 1


def test_a_failed_rescan_keeps_the_page(served: Running) -> None:
    page = served.app.page

    def broken() -> Scan:
        raise OSError("inventory gone")

    served.app.scanner = broken
    status, _, body = served.api("POST", "/api/rescan", b"{}")
    assert status == 500
    assert "inventory gone" not in body.decode()  # the reason goes to the terminal
    assert any("inventory gone" in line for line in served.log)
    assert served.request("GET", f"/?t={served.app.token}")[2] == page
    assert not served.app.lock.locked()


def test_unknown_paths_and_methods(served: Running) -> None:
    assert served.api("GET", "/api/nothing")[0] == 404
    assert served.api("POST", "/api/nothing", b"{}")[0] == 404
    assert served.api("PUT", "/api/rescan", b"{}")[0] == 501
    assert served.api("DELETE", "/")[0] == 501


def test_responses_carry_the_safety_headers(served: Running) -> None:
    for status, headers, _ in (
        served.request("GET", f"/?t={served.app.token}"),
        served.api("GET", "/api/state"),
        served.request("GET", "/"),
    ):
        assert status in (200, 403)
        assert headers["content-security-policy"] == CSP
        assert headers["x-frame-options"] == "DENY"
        assert headers["cache-control"] == "no-store"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["x-content-type-options"] == "nosniff"
        assert not any(name.startswith("access-control-") for name in headers)


def test_the_policy_allows_exactly_the_pages_scripts(served: Running) -> None:
    page = served.request("GET", f"/?t={served.app.token}")[2].decode()
    # Exactly what a browser hashes: everything between the tags.
    scripts = re.findall(r"<script>(.*?)</script>", page, re.S)
    assert len(scripts) == 2
    allowed = set(re.findall(r"'sha256-([^']+)'", CSP))
    found = {base64.b64encode(hashlib.sha256(s.encode()).digest()).decode() for s in scripts}
    assert found == allowed
    assert "unsafe-inline" not in CSP.split("script-src")[1].split(";")[0]
    assert "frame-ancestors 'none'" in CSP


def test_the_token_is_never_logged(served: Running) -> None:
    served.request("GET", f"/?t={served.app.token}")
    served.request("GET", "/?t=guess")
    served.api("POST", "/api/rescan", b"{}")
    assert served.log
    assert not any(served.app.token in line or "guess" in line for line in served.log)


def test_every_token_is_new(ctx: Context) -> None:
    first = scan(ctx)
    tokens = {App(first, lambda: first).token for _ in range(5)}
    assert len(tokens) == 5
    assert all(len(token) >= 40 for token in tokens)


def test_served_page_is_the_map_page_plus_lines_of_its_own(ctx: Context) -> None:
    found = scan(ctx)
    static = render_html(found).splitlines()
    served = render_html(found, served=True).splitlines()
    # Every line of the map's file is there, in order, and nothing else changed.
    it = iter(served)
    assert all(line in it for line in static)
    added = list(served)
    for line in static:
        added.remove(line)
    assert added == [
        _SERVED_CSS,
        '<button id="rescan" type="button">Rescan</button>',
        '<span id="served-status" class="sub" role="status"></span>',
        "<script>",
        *SCRIPTS[1].splitlines(),
        "</script>",
    ]


def test_the_command_serves_until_stopped(machine: FakeMachine) -> None:
    env = {**os.environ, "HOME": str(machine.home)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "modelmap", "serve", "--no-open", "--no-defaults",
         str(machine.home)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )  # fmt: skip
    try:
        assert proc.stdout is not None
        line = proc.stdout.readline()
        match = re.match(r"Serving the map at (http://127\.0\.0\.1:(\d+)/\?t=(\S+))$", line)
        assert match, line
        port, token = int(match.group(2)), match.group(3)
        conn = http.client.HTTPConnection(HOST, port, timeout=10)
        conn.request("GET", f"/?t={token}")
        response = conn.getresponse()
        assert response.status == 200
        assert b'id="rescan"' in response.read()
        conn.close()
    finally:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.mark.parametrize(
    "argv",
    [
        ["serve", "--redact"],
        ["serve", "--port", "70000"],
        ["serve", "--port", "x"],
        ["serve", "--from-json", "inv.json", "--only", "hf"],
    ],
)
def test_serve_usage_errors(
    ctx: Context, argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "inv.json").write_text(Scan().to_json())
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(argv, ctx=ctx)
    assert exc.value.code == 2


# ---------------------------------------------------------------- removal


@pytest.fixture
def tools(tmp_path: Path, machine: FakeMachine) -> Tools:
    store = Path(os.path.realpath(machine.ollama_store))
    return install(tmp_path / "tools", ["tiny:1b", "tiny:1b-4k", "other:latest"], store)


@pytest.fixture
def removing(machine: FakeMachine, tools: Tools) -> Iterator[Running]:
    ctx = Context(home=machine.home, env=tools.env("trash", "ollama", "hf"), platform="darwin")
    with start(ctx) as running:
        yield running


def _view(running: Running, label: str) -> str:
    (record,) = [r for r in running.app.scan.records if f"{r.provider}:{r.name}" == label]
    return record_view(record)


def _plan(running: Running, label: str) -> dict[str, Any]:
    status, _, body = running.api(
        "POST", "/api/plan", json.dumps({"view": _view(running, label)}).encode()
    )
    assert status == 200, body
    return json.loads(body)


def _remove(running: Running, view: str, digest: str, **headers: str) -> tuple[int, dict[str, Any]]:
    body = json.dumps({"view": view, "digest": digest}).encode()
    status, _, data = running.api("POST", "/api/remove", body, **headers)
    return status, json.loads(data)


def test_plans_are_the_ones_rm_makes(removing: Running, tools: Tools) -> None:
    # Same function, same inputs: the running Ollama's stores, the PATH, home.
    app = removing.app
    live = [r.resolved for r in app.scan.roots if r.provider == "ollama" and r.source != "layout"]
    for record in app.scan.records:
        expected = plan_json(plan(record, live, env=app.env, home=app.home), record_view(record))
        body = json.dumps({"view": record_view(record)}).encode()
        assert json.loads(removing.api("POST", "/api/plan", body)[2]) == expected
    actions = {
        _plan(removing, label)["action"] for label in ("generic:models/loose", "ollama:tiny:1b")
    }
    assert actions == {"trash", "tool"}


def test_removing_a_loose_file(removing: Running, machine: FakeMachine, tools: Tools) -> None:
    shown = _plan(removing, "generic:models/loose")
    assert shown["paths"] == [str(machine.files["loose"])]
    assert shown["undo"] is True and shown["command"] == ""
    status, reply = _remove(removing, shown["view"], shown["digest"])
    assert status == 200, reply
    assert reply["removed"] and "Trash" in reply["message"]
    assert not machine.files["loose"].exists()
    assert (tools.trash / "loose.gguf").exists()
    # Scanned again before and after: the page no longer has the record.
    assert len(removing.scans) == 3
    assert reply["state"]["records"] == len(removing.scans[-1].records)
    page = removing.request("GET", f"/?t={removing.app.token}")[2].decode()
    assert shown["view"] not in page
    assert any(line.startswith("removing generic:models/loose") for line in removing.log)
    assert not any(removing.app.token in line for line in removing.log)


def test_removing_an_ollama_model_names_every_tag(removing: Running, tools: Tools) -> None:
    shown = _plan(removing, "ollama:tiny:1b")
    assert shown["command"] == "ollama rm tiny:1b tiny:1b-4k"
    assert shown["variants"] == ["tiny:1b", "tiny:1b-4k"]
    assert shown["undo"] is False
    status, reply = _remove(removing, shown["view"], shown["digest"])
    assert status == 200, reply
    assert tools.tags.read_text() == "other:latest\n"


def test_a_plan_that_changed_is_not_carried_out(removing: Running, machine: FakeMachine) -> None:
    folder = machine.omlx_models / "acme" / "Sharded"
    shown = _plan(removing, "omlx:acme/Sharded")
    assert shown["paths"] == [str(folder)]
    # Between the dialog and the click, the folder gains a file of someone else's.
    (folder / "notes.txt").write_text("mine\n")
    status, reply = _remove(removing, shown["view"], shown["digest"])
    assert status == 409
    assert reply["plan"]["digest"] != shown["digest"]
    assert folder.is_dir() and (folder / "notes.txt").exists()


def test_a_wrong_digest_removes_nothing(removing: Running, machine: FakeMachine) -> None:
    shown = _plan(removing, "generic:models/loose")
    status, reply = _remove(removing, shown["view"], "0" * 64)
    assert status == 409 and reply["plan"]["digest"] == shown["digest"]
    assert machine.files["loose"].exists()


def test_a_refused_record_is_refused_by_the_server_too(removing: Running) -> None:
    caches = [r for r in removing.app.scan.records if r.kind == "cache"]
    assert caches
    view = record_view(caches[0])
    shown = json.loads(removing.api("POST", "/api/plan", json.dumps({"view": view}).encode())[2])
    assert shown["action"] == "refuse"
    status, reply = _remove(removing, view, shown["digest"])
    assert status == 409 and reply["error"] == shown["note"]


def test_a_record_already_gone(removing: Running, machine: FakeMachine) -> None:
    shown = _plan(removing, "generic:models/loose")
    machine.files["loose"].unlink()
    status, reply = _remove(removing, shown["view"], shown["digest"])
    assert status == 409 and reply["gone"] is True
    assert removing.api("POST", "/api/plan", json.dumps({"view": "rec-nothing"}).encode())[0] == 404


def test_a_failed_removal_says_why_and_redraws(removing: Running, tools: Tools) -> None:
    shown = _plan(removing, "generic:models/loose")
    tools.trash.rmdir()  # the stand-in's `mv` into the Trash now fails
    status, reply = _remove(removing, shown["view"], shown["digest"])
    assert status == 500
    assert reply["error"].startswith("Could not remove models/loose")
    assert "state" in reply
    assert any(line.startswith("failed:") for line in removing.log)


def test_one_removal_at_a_time(removing: Running, machine: FakeMachine) -> None:
    shown = _plan(removing, "generic:models/loose")
    with removing.app.lock:
        status, reply = _remove(removing, shown["view"], shown["digest"])
    assert status == 409 and "Busy" in reply["error"]
    assert machine.files["loose"].exists()


@pytest.mark.parametrize(
    "body",
    [{}, {"view": "x"}, {"digest": "x"}, {"view": 1, "digest": "x"}, {"view": "x", "digest": None}],
)
def test_removal_needs_a_view_and_a_digest(removing: Running, body: dict[str, Any]) -> None:
    assert removing.api("POST", "/api/remove", json.dumps(body).encode())[0] == 400
    assert removing.api("POST", "/api/plan", json.dumps({"view": 1}).encode())[0] == 404


def test_removal_is_refused_without_the_token_or_from_another_page(
    removing: Running, machine: FakeMachine
) -> None:
    shown = _plan(removing, "generic:models/loose")
    body = json.dumps({"view": shown["view"], "digest": shown["digest"]}).encode()
    headers = {"Content-Type": "application/json"}
    assert removing.request("POST", "/api/remove", headers=headers, body=body)[0] == 403
    assert (
        _remove(removing, shown["view"], shown["digest"], Origin="https://evil.example")[0] == 403
    )
    assert machine.files["loose"].exists()


def test_read_only_serves_no_removal(machine: FakeMachine, tools: Tools) -> None:
    ctx = Context(home=machine.home, env=tools.env("trash"), platform="darwin")
    with start(ctx, read_only=True) as running:
        page = running.request("GET", f"/?t={running.app.token}")[2].decode()
        assert 'data-remove="' not in page and 'id="rm-dialog"' not in page
        assert "The page itself deletes nothing" in page
        shown = _plan(running, "generic:models/loose")  # a plan only describes
        assert _remove(running, shown["view"], shown["digest"])[0] == 403
    assert machine.files["loose"].exists()


def test_the_page_offers_remove_only_where_rm_would(removing: Running) -> None:
    page = removing.request("GET", f"/?t={removing.app.token}")[2].decode()
    buttons = set(re.findall(r'data-remove="([^"]+)"', page))
    assert page.count('id="rm-dialog"') == 1
    app = removing.app
    live = [r.resolved for r in app.scan.roots if r.provider == "ollama" and r.source != "layout"]
    for record in app.scan.records:
        refused = plan(record, live, check_disk=False).action == "refuse"
        assert (record_view(record) in buttons) is not refused, record.name
