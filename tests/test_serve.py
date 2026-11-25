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
from pathlib import Path

import pytest

from modelmap.cli import main
from modelmap.providers.base import Context
from modelmap.records import Scan
from modelmap.render_html import _SERVED_CSS, SCRIPTS, render_html
from modelmap.scanner import scan
from modelmap.serve import CSP, HOST, TOKEN_HEADER, App, Server, make_server
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


@pytest.fixture
def served(ctx: Context) -> Iterator[Running]:
    scans = [scan(ctx)]

    def rescan() -> Scan:
        scans.append(scan(ctx))
        return scans[-1]

    log: list[str] = []
    server = make_server(App(scans[0], rescan), 0, log.append)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield Running(server, scans, log)
    server.shutdown()
    server.server_close()
    thread.join()


def test_listens_on_loopback_only(served: Running) -> None:
    assert served.server.server_address[0] == "127.0.0.1"
    assert served.server.url == f"http://127.0.0.1:{served.server.port}/?t={served.app.token}"


def test_the_page_is_the_map_drawn_for_serving(served: Running) -> None:
    status, headers, body = served.request("GET", f"/?t={served.app.token}")
    assert status == 200
    assert headers["content-type"] == "text/html; charset=utf-8"
    assert body.decode() == render_html(served.scans[0], served=True)
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
    assert page == render_html(served.scans[1], served=True)


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
