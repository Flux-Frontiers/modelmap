"""``modelmap serve``: the map from a web server on this machine.

The server listens on 127.0.0.1 only. Every request must name the server by
the address it listens on (``Host``), may come only from the server's own
page (``Origin``), and must carry the run's token: in the address for the
page, in the ``X-Modelmap-Token`` header for the API. A page on another site
can therefore neither read the map nor make the server act, even though the
server is reachable from the browser it runs in.

Routes:

- ``GET /?t=TOKEN``: the page, drawn by :func:`render_html` with ``served``.
- ``GET /api/state``: when the scan ran and what it found, in totals.
- ``POST /api/rescan``: scan again and redraw the page.

The page is drawn once per scan and kept, so loading it does not scan.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import parse_qs, urlsplit

from modelmap.records import Scan
from modelmap.render_html import SCRIPTS, render_html

HOST = "127.0.0.1"
TOKEN_HEADER = "X-Modelmap-Token"
MAX_BODY = 4096

# A browser hashes everything between <script> and </script>, and the page
# puts each script on lines of its own: the newlines are part of the text.
_HASHES = " ".join(
    "'sha256-" + base64.b64encode(hashlib.sha256(f"\n{script}\n".encode()).digest()).decode() + "'"
    for script in SCRIPTS
)
#: Inline scripts run only if they are the page's own; nothing loads from
#: elsewhere, the page cannot be framed, and it calls only this server.
CSP = (
    f"default-src 'none'; script-src {_HASHES}; style-src 'unsafe-inline'; img-src data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)


class App:
    """What every request shares: the scan, its page, the token and a lock.

    :param scan: The first scan, already made.
    :param scanner: Makes a new scan; called by a rescan.
    :param use_disk: Size tiles by bytes allocated on disk.
    :param token: The run's token; a random one by default.
    """

    def __init__(
        self,
        scan: Scan,
        scanner: Callable[[], Scan],
        *,
        use_disk: bool = False,
        token: str | None = None,
    ) -> None:
        self.scanner = scanner
        self.use_disk = use_disk
        self.token = token or secrets.token_urlsafe(32)
        #: Held while the scan changes; a second request is told to wait.
        self.lock = threading.Lock()
        self._set(scan)

    def _set(self, scan: Scan) -> None:
        page = render_html(scan, use_disk=self.use_disk, served=True).encode("utf-8")
        self.scan, self.page, self.scanned_at = scan, page, time.time()

    def rescan(self) -> None:
        """Scan again and redraw. The caller holds :attr:`lock`."""
        self._set(self.scanner())

    def state(self) -> dict[str, Any]:
        """The scan in totals, as the API returns it."""
        scan = self.scan
        return {
            "scanned_at": round(self.scanned_at, 3),
            "elapsed_s": scan.elapsed_s,
            "records": len(scan.records),
            "size_bytes": scan.size_bytes,
            "disk_bytes": scan.disk_bytes,
            "warnings": len(scan.warnings),
        }

    def token_ok(self, given: str) -> bool:
        return hmac.compare_digest(given.encode("utf-8"), self.token.encode("utf-8"))


class Server(ThreadingHTTPServer):
    """The HTTP server; :func:`make_server` builds it."""

    daemon_threads = True
    app: App
    log: Callable[[str], None]

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def hosts(self) -> tuple[str, str]:
        return (f"{HOST}:{self.port}", f"localhost:{self.port}")

    @property
    def url(self) -> str:
        """The page's address, token included."""
        return f"http://{HOST}:{self.port}/?t={self.app.token}"


class Handler(BaseHTTPRequestHandler):
    server_version = "modelmap"
    sys_version = ""

    @property
    def owner(self) -> Server:
        return cast(Server, self.server)

    # Logging. The request line holds the token, so it is never logged.
    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        self.owner.log(f"{self.command} {urlsplit(self.path).path} {code}")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (base class name)
        self.owner.log(format % args)

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy", CSP)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, value: dict[str, Any]) -> None:
        self._send(code, json.dumps(value).encode("utf-8"), "application/json")

    def _error(self, code: int, message: str) -> None:
        self._json(code, {"error": message})

    def _allowed(self) -> bool:
        """Host and Origin checks; answers the request when they fail."""
        if self.headers.get("Host", "") not in self.owner.hosts:
            self._error(403, "Use the address modelmap serve printed.")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin not in {f"http://{h}" for h in self.owner.hosts}:
            self._error(403, "Requests from other pages are refused.")
            return False
        return True

    def do_GET(self) -> None:  # noqa: N802 (base class name)
        if not self._allowed():
            return
        url = urlsplit(self.path)
        app = self.owner.app
        if url.path == "/":
            if not app.token_ok(parse_qs(url.query).get("t", [""])[0]):
                self._error(403, "Open the address modelmap serve printed; it holds the token.")
                return
            self._send(200, app.page, "text/html; charset=utf-8")
        elif url.path == "/api/state":
            if not app.token_ok(self.headers.get(TOKEN_HEADER, "")):
                self._error(403, "Missing or wrong token.")
                return
            self._json(200, app.state())
        else:
            self._error(404, "Not found.")

    def do_POST(self) -> None:  # noqa: N802 (base class name)
        if not self._allowed():
            return
        app = self.owner.app
        if not app.token_ok(self.headers.get(TOKEN_HEADER, "")):
            self._error(403, "Missing or wrong token.")
            return
        if self.headers.get_content_type() != "application/json":
            self._error(415, "Send JSON.")
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(411, "Content-Length is required.")
            return
        if not 0 <= length <= MAX_BODY:
            self._error(413, "Request too large.")
            return
        try:
            body = json.loads(self.rfile.read(length) or b"null")
        except ValueError:
            body = None
        if not isinstance(body, dict):
            self._error(400, "Send a JSON object.")
            return

        if urlsplit(self.path).path != "/api/rescan":
            self._error(404, "Not found.")
            return
        if not app.lock.acquire(blocking=False):
            self._error(409, "Busy: a scan is running. Try again when it ends.")
            return
        try:
            app.rescan()
        except (OSError, ValueError) as exc:  # an inventory file gone or broken; the old page stays
            self.owner.log(f"rescan failed: {exc!r}")
            self._error(500, "The scan failed; the terminal running modelmap serve says why.")
            return
        finally:
            app.lock.release()
        self._json(200, app.state())


def make_server(app: App, port: int = 0, log: Callable[[str], None] | None = None) -> Server:
    """Bind the server to 127.0.0.1; ``serve_forever`` starts it.

    :param app: The shared state.
    :param port: Port to listen on; 0 takes a free one.
    :param log: Receives one line per request; silent by default.
    :raises OSError: When the port cannot be bound.
    """
    server = Server((HOST, port), Handler)
    server.app = app
    server.log = log or (lambda line: None)
    return server
