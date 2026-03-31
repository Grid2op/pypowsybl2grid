# pypowsybl2grid

An integration between [Grid2op](https://github.com/rte-france/Grid2Op) and [PyPowSybl](https://pypowsybl.readthedocs.io/).

**License:** MPL-2.0  
**Python:** >= 3.9

---

## Installation

To install the package as a standard user, you can use `pip` or `uv`. 
Note that to fully utilize the integration, you may want to install the optional `pypowsybl` dependency:

```bash
uv sync --extras pypowsybl
```

or 

```bash
pip install pypowsybl2grid[pypowsybl]
```

## Development setup

To set up the development environment with virtual environment, linting, formatting, and testing:

Create a virtual environment using uv:

```bash
uv venv
```

Add the package to the virtual environment:

```bash
uv sync --all-extras
```

Setup pre-commit hooks for linting and formatting:

```bash
uv run prek install
```

When you want to run tests, you can use:

```bash
uv run pytest
```

When commiting code, the pre-commit hooks will automatically check for linting and formatting issues. Don't forget to sign your commits with `-s`

```bash
git commit -s -m "Your commit message"
```
