#!/usr/bin/env bash
# Stage the Python the app runs: a standalone CPython with modelmap installed.
#
# Writes macos/Python/, which the Xcode project copies into
# ModelMap.app/Contents/Resources/Python. The interpreter is uv's managed
# CPython (a python-build-standalone build): it finds libpython relative to
# its own executable, so the folder works wherever it is copied. modelmap
# needs only the standard library, so the wheel built from this checkout is
# the only package installed, and pip is removed afterward.
#
# Usage: scripts/stage_python.sh   (from macos/; `make python` runs it)

set -euo pipefail

PY_VERSION="${PY_VERSION:-3.12}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
OUT="$HERE/Python"
WHEELS="$HERE/build/wheel"

# Only a uv-managed build will do: a Homebrew or python.org Python links its
# framework by absolute path and breaks once copied into the app.
export UV_PYTHON_PREFERENCE=only-managed
uv python install "$PY_VERSION" >/dev/null
SRC="$(dirname "$(dirname "$(uv python find "$PY_VERSION")")")"
SRC="$(cd "$SRC" && pwd -P)"   # uv's short name is a link to the versioned folder
if ! otool -L "$SRC/bin/python$PY_VERSION" | rg -q '@executable_path/../lib/libpython'; then
  echo "error: $SRC is not relocatable (libpython not found relative to the executable)" >&2
  exit 1
fi

rm -rf "$OUT" "$WHEELS"
mkdir -p "$WHEELS"
cp -R "$SRC" "$OUT"

(cd "$REPO" && poetry build -q -f wheel -o "$WHEELS")
# uv marks its Pythons externally managed (PEP 668); this copy belongs to the app.
rm -f "$OUT/lib/python$PY_VERSION/EXTERNALLY-MANAGED"
"$OUT/bin/python3" -I -m pip install -q --no-deps --no-index --no-compile \
  --disable-pip-version-check "$WHEELS"/modelmap-*.whl

# Nothing modelmap imports: the Tk GUI toolkit, IDLE, the test suite, pip
# itself, and headers for building extensions.
LIB="$OUT/lib/python$PY_VERSION"
rm -rf "$LIB"/{test,idlelib,tkinter,turtledemo,ensurepip,lib2to3} \
  "$LIB"/site-packages/pip "$LIB"/site-packages/pip-*.dist-info \
  "$LIB"/lib-dynload/_tkinter* \
  "$OUT"/lib/{tcl*,tk*,itcl*,thread*,libtcl*,libtk*,pkgconfig} \
  "$OUT"/include "$OUT"/share "$OUT"/bin/{idle*,pip*,pydoc*,*-config}
find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
"$OUT/bin/python3" -I -m compileall -q -j 0 "$LIB" >/dev/null || true

VERSION="$("$OUT/bin/python3" -I -m modelmap --version)"
echo "Staged $VERSION on Python $("$OUT/bin/python3" -I -c 'import platform; print(platform.python_version())') ($(du -sh "$OUT" | cut -f1)) in $OUT"
