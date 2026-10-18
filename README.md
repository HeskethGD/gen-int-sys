# Generating integrable systems from an autonomous ordinary differential equation

Manuscript and computational companion by Graham Hesketh.

The paper develops a scalar-equation-first construction within a Kummer-type family of canonical Hamiltonian systems. It covers modal-power reduction, reconstruction of individual modes, logarithmic and rational generating equations, incomplete beta inversion, and examples based on inverse spherical Bessel and modular lambda functions.

The computational companion consists of eight self-contained Marimo notebooks, with executed HTML walkthroughs and symbolic and numerical tests.

## Contents

- `latex/main.tex`: manuscript source.
- `latex/main.pdf`: compiled manuscript snapshot.
- `latex/references.bib`: bibliography.
- `latex/figures/`: manuscript figures.
- `code/`: eight self-contained Marimo notebooks; see the [notebook guide](code/README.md).
- `code/exports/`: executed HTML walkthroughs readable without Python.
- `tests/`: symbolic identities, exact examples, branch checks and independent numerical validation; see [verification scope](tests/README.md).
- `requirements.txt`, `requirements-lock.txt`, `environment.json`: direct dependencies, the installed-package snapshot and tested environment.

## Read and rebuild the manuscript

Read [the supplied PDF](latex/main.pdf) without installing Python or LaTeX.

To rebuild, install a LaTeX distribution providing `latexmk`, pdfLaTeX and Biber, then run from this repository root:

```sh
latexmk -interaction=nonstopmode -halt-on-error latex/main.tex
```

The new output is `latex/build/main.pdf`; auxiliary files are also confined to `latex/build/`. To refresh the supplied snapshot after a successful build:

```sh
cp latex/build/main.pdf latex/main.pdf
```

## Computational companion

The notebooks and tests were validated with Python 3.13.4. From this repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m marimo edit code/01_general_hamiltonian.py
```

`requirements.txt` lists the directly used packages; `requirements-lock.txt` records all installed package versions in the tested environment. The latter is an environment snapshot, not a guarantee of compatibility with every Python version or platform.

Run the tests and validate the notebook structure:

```sh
python -m pytest -q
python -m marimo check code/*.py
```

To execute and export a notebook:

```sh
python -m marimo export html code/01_general_hamiltonian.py -o code/exports/01_general_hamiltonian.html --force
```

Run each notebook independently in the same way. Each notebook defines its own functions and requires no other notebook.

The notebook sources show the derivations. Tests check independent formulations and numerical reconstructions. See [verification scope and numerical accuracy](tests/README.md) for the checks and their limits; the executed notebooks and manuscript captions also give integration settings.

## License

The software and accompanying software documentation use the [MIT license](LICENSE), with the exact scope stated in [LICENSE_SCOPE.md](LICENSE_SCOPE.md). The manuscript, figures, generated notebook exports and numerical reports are excluded from this software license. Existing licenses previously granted for distributed material are unaffected. Third-party dependencies retain their own licenses.
