# anima-def-attackvla

Defense-oriented ANIMA scaffold for AttackVLA (arXiv:2511.12149).

## Quick start
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv sync --extra dev
uv run pytest -q
uv run uvicorn anima_def_attackvla.serve:app --reload
```
