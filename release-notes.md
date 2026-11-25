# Release Notes: v0.2.0

> Released: 2026-11-25

modelmap can now remove models from the map itself. `modelmap serve`
scans, serves the map from your own machine, and gives each model a
Remove button that does exactly what `modelmap rm` does: it shows every
command or file that would go, waits for you to confirm, and checks
again before acting.

## What changed

**The map as a local app.** `modelmap serve` opens the same map in your
browser, served from `127.0.0.1`, with a Rescan button and a Remove
button on each model. The Remove dialog lists the command or every path,
any Ollama tags that go with the model, and whether the removal can be
undone from the Trash. When you confirm, the server scans and plans
again; if anything has changed since the dialog opened, it removes
nothing and shows the new list. After a removal the map is redrawn from
a fresh scan.

**Built to be safe on a shared browser.** The server listens only on
this machine, answers only to its own page, and needs a token made for
each run, which it never writes into the page or the log. A page on
another site cannot read the map or ask it to remove anything. Serving an
inventory from another machine, or adding `--read-only`, turns removal
off.

**A face and a home.** modelmap has its own Art Techno logo, a treemap
drawn as a chip floorplan, and every page opens with a banner showing the
version and a link to Flux-Frontiers. The documentation is now a site at
https://flux-frontiers.github.io/modelmap/, with the guide, the command
reference, the cheat sheet and an API reference.

**A fix for repeated scans.** A second scan in the same process
undercounted, because the first scan's bookkeeping carried over. The
command line scans once, so earlier results were right; the server
rescans, and now each scan starts fresh.

## Upgrading

Install or upgrade from PyPI with `pipx install modelmap` (or
`pip install -U modelmap`). Nothing needs migrating; inventories made by
0.1.0 still load. Try `modelmap serve`, or `modelmap serve --read-only`
to look without the Remove buttons.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
