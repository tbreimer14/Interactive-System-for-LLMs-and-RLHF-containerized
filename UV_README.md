# UV Project Setup & Usage Guide
# last update: (05.04.2026 15:54pm)

This project uses **`uv`** — a fast, modern Python package manager and project manager written in Rust.

## What is `uv`?

`uv` replaces `pip`, `pip-tools`, `venv`, and `virtualenv`. It's:
- **Fast** — 10-100x faster than pip
- **Simple** — Minimal commands for common tasks
- **Reliable** — Deterministic lock file (`uv.lock`)
- **Modern** — Works with `pyproject.toml`, Python 3.11+

---

## Installation

### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then restart your terminal or run:
```
$PROFILE
```

Verify installation:
```bash
uv --version
```

---

## Initial Setup

Run these commands once in the project root:

```bash
# Create a virtual environment in .venv/
uv venv

# Install all dependencies from pyproject.toml
uv sync
```

**That's it!** Your environment is ready.

---

## Common Commands

### 🔧 Environment Management

| Command | Purpose |
|---------|---------|
| `uv venv` | Create a virtual environment |
| `uv sync` | Install/update all dependencies from lock file |
| `uv python list` | List available Python versions |
| `uv python install 3.11` | Download a specific Python version |

### 📦 Dependency Management

| Command | Purpose |
|---------|---------|
| `uv add numpy` | Add a new package |
| `uv add --dev pytest` | Add a dev-only package |
| `uv remove numpy` | Remove a package |
| `uv pip list` | Show installed packages |
| `uv pip freeze` | Show all packages with versions |

### ▶️ Running Code

| Command | Purpose |
|---------|---------|
| `uv run python script.py` | Run a Python script |
| `uv run python --version` | Check Python version |
| `uv run pytest` | Run tests |
| `uv run pytest tests/test_rag.py` | Run specific test file |
| `uv run python -m fastapi.run app:app` | Run FastAPI app |

### 🔍 Inspection

| Command | Purpose |
|---------|---------|
| `uv pip show numpy` | Show details about a package |
| `uv pip compile requirements.txt` | Generate lock file |
| `.venv/Scripts/activate` (Windows) | Activate venv directly (not recommended with uv) |

---

## Workflow Examples

### Adding a New Package

```bash
# Add to production dependencies
uv add requests

# Add to dev dependencies (testing, linting, etc)
uv add --dev black

# Add a specific version
uv add "numpy>=1.20,<2.0"
```

The command automatically updates `pyproject.toml` and `uv.lock`.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_model.py::test_inference

# Run and show coverage
uv run pytest --cov=backend
```

### Running Your Application

```bash
# Start FastAPI server
uv run python -m uvicorn backend.app:app --reload

# Run a training script
uv run python backend/scripts/train_model.py

# Run a notebook
uv run jupyter notebook
```

### Updating All Dependencies

```bash
# Update lock file with latest compatible versions
uv sync --upgrade

# Update only one package
uv add --upgrade numpy
```

---

## Project Structure

```
llm-rlhf-system-va/
├── backend/                   # Your app code
│   ├── rag_system/
│   ├── reward_model/
│   └── app.py
├── tests/                     # Test files
│   └── test_rag.py
├── pyproject.toml            # Project config & dependencies
├── uv.lock                   # Lock file (auto-generated, commit this)
├── .python-version           # Python version (3.11)
├── .venv/                    # Virtual environment (auto-generated, gitignored)
├── README.md                 # Project overview
└── UV_README.md              # This file
```

---

## Tips & Best Practices

### ✅ Do This
- Run all Python code with `uv run` (keeps environment clean)
- Commit `uv.lock` to version control (ensures reproducible builds)
- Use `uv add --dev` for testing/dev tools
- Check `pyproject.toml` before running `uv sync`

### ❌ Don't Do This
- Don't manually edit `.venv/` (let `uv` manage it)
- Don't use `pip` directly (use `uv add` instead)
- Don't activate `.venv/` manually — just use `uv run`
- Don't ignore `uv.lock` in git

### When Stuck
```bash
# Start fresh
uv sync --fresh

# Remove and recreate venv
rm -r .venv
uv venv
uv sync

# Check what uv is doing
uv --verbose sync
```

---

## File Reference

### `pyproject.toml`
Modern Python project configuration. Defines:
- Project metadata (name, version, description)
- Dependencies (what packages you need)
- Development dependencies
- Tool configuration (pytest, ruff, etc.)

### `uv.lock`
Auto-generated lock file with exact versions of all dependencies. **Always commit this to git.**

### `.python-version`
Specifies which Python version to use (3.11). `uv` reads this automatically.

### `.venv/`
Your virtual environment directory. Gitignored automatically.

---

## Troubleshooting

### "uv command not found"
Reinstall `uv` using the installation command above. Restart terminal after install.

### "ModuleNotFoundError: No module named 'X'"
Run `uv sync` to install dependencies:
```bash
uv sync
```

### "Python version mismatch"
Ensure your system has Python 3.11:
```bash
uv python install 3.11
```

### "Lock file out of sync"
Update the lock file:
```bash
uv sync --upgrade
```

---

## Next Steps

1. ✅ Run `uv venv` and `uv sync`
2. ✅ Verify with `uv run python --version`
3. Start building features:
   - `uv add` new packages as needed
   - `uv run pytest` while developing
   - `uv run python` to test scripts

## Resources

- Official Docs: https://docs.astral.sh/uv/
- Getting Started: https://docs.astral.sh/uv/getting-started/
- Python Semver: https://docs.astral.sh/uv/guide/versions/
