# dnamic-toolkit
Contains common helper functions, physics calculations, for use in DNAMIC (c)ontrolled labs.

## Installing UV (the modern replacement for conda, pip, virtualenv, piptools,...)

Follow Astral’s install guide: https://docs.astral.sh/uv/getting-started/installation/ 

## Using `dnamic-toolkit` from another `uv` project without installing editably

From your *other* project directory:

```bash
uv add "dnamic-toolkit @ git+https://github.com/CornishLabs/dnamic-toolkit.git"
uv sync
```

For having an install, and being able to edit it, see below.

## Quickstart to developing + using simultaneously

### Clone + create the project environment
```bash
git clone https://github.com/CornishLabs/dnamic-toolkit.git
cd dnamic-toolkit
uv sync # This updates the venv associated with this folder
```

This creates/updates the project’s `.venv` and installs the project in editable mode in this project venv for development.

### Run the tests

```bash
uv run pytest
```

`uv run` executes commands inside the project environment (it syncs before if necessary).

### Try a quick import

```bash
uv run python -c "import dnamic_toolkit; print('import ok')"
```

(`src/` contains the package and `tests/` contains the test suite.)

### Run an example

```bash
uv run python examples/<example_file>.py
```

(See the `examples/` folder for runnable scripts.)

---

## Use this editable install in another project setup with UV
```bash
uv add --editable /path/to/cloned/dnamic-toolkit
uv sync
```

---

## Notes for contributors

* Add runtime deps:

  ```bash
  uv add <package>
  ```

* Add dev deps (tests/lint tooling). The `dev` group is installed by default:

  ```bash
  uv add --dev pytest
  ```

* CI/repro builds: fail if `uv.lock` would change:

  ```bash
  uv sync --locked
  ```

  or

  ```bash
  uv run --locked pytest
  ```
