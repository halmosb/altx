# Development

This page explains how to run the test suite, configure pre-commit hooks, and build the documentation locally.

For environment setup and installation instructions see the [Installation](installation.md) page.

---

## Running the tests

The test suite uses **pytest** with coverage support. The full configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`.

### Run all tests

```bash
python -m pytest
```

pytest is configured with `-v` (verbose), `--strict-markers`, and `--disable-warnings` by default, so you will see one line per test case.

### Run a single test

Specify the file, class, and method separated by `::`:

```bash
python -m pytest tests/test_altx.py::TestTrain::test_train_populates_ps_for_all_rlk
```

### Run with coverage

```bash
python -m pytest --cov
```

Coverage is measured over the `altx/` package (configured under `[tool.coverage.run]`). The `tests/` directory is excluded from the report.

### Test structure

| File | What it tests |
|------|--------------|
| `tests/test_altx.py` | All public methods of `Altx` — init validation, embed, train, save/load, multiply, transform, transform_set, save features, header generation, print_number_of_laws |
| `tests/test_extract_methods.py` | All extraction methods in `ExtractMethods` — excess_kurtosis, nth_moment, and the main extract dispatcher |

---

## Pre-commit hooks

Pre-commit hooks run automatically on every `git commit` and enforce code quality. They are configured in `.pre-commit-config.yaml`.

### Install the hooks

Run this once after cloning the repository:

```bash
pre-commit install
```

After that, `git commit` will trigger all hooks automatically on the staged files.

### Run hooks manually

On specific files:

```bash
pre-commit run --files altx/altx.py altx/extract_methods.py
```

On the entire repository:

```bash
pre-commit run --all-files
```

### Hooks that are configured

| Hook | Tool | What it checks / fixes |
|------|------|------------------------|
| `trailing-whitespace` | pre-commit-hooks | Trims trailing whitespace from every line |
| `end-of-file-fixer` | pre-commit-hooks | Ensures files end with exactly one newline |
| `check-yaml` | pre-commit-hooks | Validates YAML syntax |
| `check-json` | pre-commit-hooks | Validates JSON syntax |
| `check-merge-conflict` | pre-commit-hooks | Blocks committing unresolved merge conflict markers |
| `check-added-large-files` | pre-commit-hooks | Rejects files larger than 5 MB |
| `no-commit-to-branch` | pre-commit-hooks | Prevents direct commits to `main` or `master` |
| `black` | Black | Auto-formats Python files to 88-character lines |
| `nbqa-black` | nbQA + Black | Applies Black formatting to Jupyter notebooks |
| `ruff-format` | Ruff | Applies Ruff's formatter (compatible with Black) |
| `ruff-check` | Ruff | Lints and auto-fixes import ordering, bugbear rules, docstring style, and more (see below) |
| `nbqa-ruff` | nbQA + Ruff | Applies Ruff linting to Jupyter notebooks |
| `mypy` | Mypy | Strict type checking (Python 3.12 target, `strict = true`), production code only |
| `pydocstyle` | pydocstyle | Enforces NumPy docstring convention |

### Ruff rule groups

The following Ruff rule groups are active (see `[tool.ruff.lint]` in `pyproject.toml`):

| Code | Group | Purpose |
|------|-------|---------|
| `E` | pycodestyle | Style errors (line length, whitespace, etc.) |
| `F` | pyflakes | Undefined names, unused imports |
| `I` | isort | Import ordering |
| `N` | pep8-naming | Class, function, and variable naming |
| `D` | pydocstyle | Docstring presence and formatting |
| `UP` | pyupgrade | Modernise Python syntax |
| `B` | flake8-bugbear | Common bugs and design issues |
| `C4` | flake8-comprehensions | Idiomatic comprehensions |
| `SIM` | flake8-simplify | Unnecessarily complex code |
| `RUF` | Ruff-specific | Ruff's own additional rules |

Ignored rules: `E741` (ambiguous names), `N802/N803/N806/N814` (non-lowercase identifiers and camelcase constants). Test files additionally ignore all `D` rules.

### Branch protection

The `no-commit-to-branch` hook blocks direct commits to `main` and `master`. Always create a feature branch first:

```bash
git switch -c feature/my-change
```

---

## Building the documentation

The documentation is built with **Sphinx** using the `pydata_sphinx_theme`. The source files live in `docs/source/` and the built HTML is written to `docs/_build/html/`.

### Install documentation dependencies

```bash
pip install -e ".[docs]"
```

This installs:

| Package | Purpose |
|---------|---------|
| `sphinx` | Core documentation builder |
| `pydata-sphinx-theme` | HTML theme |
| `myst-parser` | Allows writing documentation pages in Markdown (`.md`) |
| `sphinx-autodoc-typehints` | Renders Python type annotations in the API reference |
| `numpydoc` | NumPy docstring support |
| `sphinx-copybutton` | Adds a copy button to code blocks |
| `sphinx-design` | Additional layout components (grids, cards, tabs) |

### Build HTML locally

The quickest way is to use the `Makefile` that ships in the `docs/` directory:

```bash
cd docs
make html
```

The output is written to `docs/build/html/`. Open `docs/build/html/index.html` in a browser to view the result. Run `make clean` first if you want to discard the cached build and start from scratch.

Alternatively, call `sphinx-build` directly from the repository root (this is what CI uses and writes to a different output directory):

```bash
sphinx-build -b html docs/source docs/_build/html
```

To treat all warnings as errors (mirrors CI behaviour):

```bash
sphinx-build -W -T -b html docs/source docs/_build/html
```

### Documentation structure

```
docs/
├── source/
│   ├── conf.py          ← Sphinx configuration (theme, extensions, version)
│   ├── index.rst        ← Top-level table of contents
│   ├── installation.md  ← Installation instructions
│   ├── usage.md         ← Usage guide and examples
│   ├── development.md   ← This page
│   └── api/
│       ├── modules.rst  ← Auto-generated module list
│       └── altx.rst     ← API reference (autodoc directives)
└── _build/
    └── html/            ← Generated HTML output (not committed)
```

`conf.py` derives the package version from the latest git tag via `setuptools-scm`. If no tag is found, it falls back to the installed package version, and then to `"0.0.0"`.

### Continuous integration

The documentation is built automatically on every push via the `.github/workflows/build-docs.yml` workflow. The workflow:

1. Builds the HTML documentation.
2. Uploads it as a GitHub Actions artifact (`html-docs`).
3. On pushes to `master` or version tags (`v*`), deploys the artifact to GitHub Pages via the `peaceiris/actions-gh-pages` action.

---

## Continuous integration overview

| Workflow | File | Trigger | What it does |
|----------|------|---------|--------------|
| Python tests | `tests.yml` | Every push and PR | Runs pytest with coverage on Python 3.12, 3.13, and 3.14; uploads a coverage badge on pushes to `halmosb/altx` |
| Pre-commit check | `pre-commit.yml` | Every push and PR | Runs all pre-commit hooks (`--all-files`) except `no-commit-to-branch` |
| Build and Deploy Docs | `build-docs.yml` | Every push, PRs to main/master | Builds Sphinx HTML; deploys to GitHub Pages on `master` or version tags |

All CI jobs run in the custom Docker image `ghcr.io/halmosb/docker-builder/python:3.14-cpu-v0.3.4`, which pre-installs the required Python version and PyTorch CPU build, see [github.com/halmosb/docker-builder](https://github.com/halmosb/docker-builder) for more details.
