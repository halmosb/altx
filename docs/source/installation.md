# Installation

## Requirements

`altx` requires **Python 3.12 or later** and the following packages:

| Package | Minimum version |
|---------|----------------|
| `numpy` | 2.0 |
| `pandas` | 2.0 |
| `torch` | 2.12 |
| `matplotlib` | 3.10 |
| `pre-commit` | 3.7 |

---

## Installing from GitHub

### Clone and install in editable mode

This is the recommended approach for developers or for users who want to stay on the latest commit:

```bash
git clone https://github.com/halmosb/altx.git
cd altx
pip install -e .
```

The `-e` flag installs the package in *editable* mode: changes you make to the source files are reflected immediately without reinstalling.

To also install the documentation build dependencies:

```bash
pip install -e ".[docs]"
```

To run the bundled examples (requires the `aeon` dataset loader):

```bash
pip install -e ".[examples]"
```

---

### Install directly from GitHub with pip

If you do not need to modify the source, you can install the latest version of the `main` branch directly:

```bash
pip install git+https://github.com/halmosb/altx.git
```

To include the optional example dependencies:

```bash
pip install "git+https://github.com/halmosb/altx.git#egg=altx[examples]"
```

To include the documentation build dependencies:

```bash
pip install "git+https://github.com/halmosb/altx.git#egg=altx[docs]"
```

---

## Conda environment

If you use conda, you can create a dedicated environment first and install into it:

```bash
conda create -n altx-env python=3.14
conda activate altx-env
pip install git+https://github.com/halmosb/altx.git
```

The development environment used in this project is called `altx-env` (Python 3.14). All commands in this documentation assume this environment is activated, or that you have an equivalent environment with the required packages.

---

## uv environment

[uv](https://docs.astral.sh/uv/) is a fast Python package and project manager written in Rust. It can download and manage Python interpreters itself, so no separate Python installation is required.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify the installation:

```bash
uv --version
```

### Create a Python 3.14 virtual environment

```bash
git clone https://github.com/halmosb/altx.git
cd altx
uv venv --python 3.14
```

uv will automatically download Python 3.14 if it is not already available on your system. The virtual environment is created in `.venv/`.

Activate it:

```bash
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### Install the package

```bash
uv pip install -e .
```

With documentation dependencies:

```bash
uv pip install -e ".[docs]"
```

With example dependencies:

```bash
uv pip install -e ".[examples]"
```

### Install directly from GitHub (no clone)

```bash
uv pip install "git+https://github.com/halmosb/altx.git"
```

---

## Verifying the installation

After installation, you can verify it works by running:

```python
import altx
from altx import ALT, ExtractMethods
print("altx imported successfully")
```

Or run the test suite from the repository root:

```bash
conda run -n altx-env python -m pytest
```

All tests should pass.
