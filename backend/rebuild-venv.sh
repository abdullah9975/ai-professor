#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Rebuild the ai-professor backend virtualenv OUTSIDE OneDrive.
#
# Why: the old venv lived at backend/venv inside a OneDrive-synced folder.
# OneDrive dehydrated and conflict-renamed files inside it, which silently
# deleted real modules (click/core.py, openai/__init__.py, numpy/_core/numeric.py,
# anyio/_core/__init__.py and others). A venv must never live in a synced folder.
#
# Run from Git Bash:   bash backend/rebuild-venv.sh
# ---------------------------------------------------------------------------
set -euo pipefail

BASE_PY="/c/Python314/python.exe"     # matches the old venv's pyvenv.cfg
VENV="/c/venvs/ai-professor"          # outside OneDrive, outside the repo
REQ="$(cd "$(dirname "$0")" && pwd)/requirements.txt"

echo "==> checking base interpreter"
if [ ! -x "$BASE_PY" ]; then
  echo "ERROR: $BASE_PY not found."
  echo "       Find yours with:  py -0p     then edit BASE_PY at the top of this script."
  exit 1
fi
"$BASE_PY" --version

if [ -e "$VENV" ]; then
  echo "==> $VENV already exists; moving it aside"
  mv "$VENV" "${VENV}.old.$(date +%s)"
fi

echo "==> creating fresh venv at $VENV"
mkdir -p /c/venvs
"$BASE_PY" -m venv "$VENV"

PY="$VENV/Scripts/python.exe"

echo "==> upgrading pip / wheel"
"$PY" -m pip install --upgrade pip wheel

echo "==> installing pinned requirements (takes a few minutes)"
"$PY" -m pip install -r "$REQ"

echo "==> verifying the packages that were corrupted, plus the app's imports"
"$PY" - <<'PY'
import importlib
mods = ["click", "click.core", "anyio", "anyio._core", "openai", "numpy",
        "fastapi", "uvicorn", "pydantic", "starlette", "supabase", "celery",
        "langchain_core", "langgraph", "fitz", "PIL", "pytesseract", "dotenv"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
        print("  OK   ", m)
    except Exception as e:
        bad.append(m)
        print("  FAIL ", m, "->", type(e).__name__, e)
if bad:
    print("\nStill broken:", ", ".join(bad))
    raise SystemExit(1)
print("\nAll imports clean.")
PY

cat <<EOF

==> Done.

Start the backend with:

    cd backend
    source $VENV/Scripts/activate
    uvicorn main:app --reload --port 8000

The old broken venv is still at backend/venv. Once the above works, delete it
to stop OneDrive syncing ~1GB of dead files:

    rm -rf backend/venv
EOF
