# dnamic-toolkit
Contains common helper functions, physics calculations, for use in DNAMIC (c)ontrolled labs.

## Quickstart (using `uv`)

### 1) Install `uv`
Follow Astral’s install guide: https://docs.astral.sh/uv/getting-started/installation/ 

### 2) Clone + create the project environment
```bash
git clone https://github.com/CornishLabs/dnamic-toolkit.git
cd dnamic-toolkit
uv sync
````

This creates/updates the project’s `.venv` and installs the project in editable mode for development.

### 3) Run the tests

```bash
uv run pytest
```

`uv run` executes commands inside the project environment (it syncs before if necessary).

### 4) Try a quick import

```bash
uv run python -c "import dnamic_toolkit; print('import ok')"
```

(`src/` contains the package and `tests/` contains the test suite.)

### 5) Run an example

```bash
uv run python examples/<example_file>.py
```

(See the `examples/` folder for runnable scripts.)

---

## Using `dnamic-toolkit` from another `uv` project

From your other project directory:

```bash
uv add "dnamic-toolkit @ git+https://github.com/CornishLabs/dnamic-toolkit.git"
uv sync
```

or if you would like it to be editable:
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
