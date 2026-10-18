# Computational notebooks

Eight self-contained Marimo notebooks accompany the manuscript:

1. `01_general_hamiltonian.py` — canonical equations, power invariants, scalar reduction, reconstruction and initial compatibility.
2. `02_inverse_construction.py` — prescribed scalar ODEs, admissible square roots, time reparameterisation and equivalent mixing functions.
3. `03_total_phase.py` — prescribed phases, correction quadratures and preservation of weighted products.
4. `04_algebraic_examples.py` — selected summary-table checks, scalar degree/genus and the Fermat/incomplete-beta family.
5. `05_log_augmented.py` — the paper's first- and second-power logarithmic examples and conservation laws.
6. `06_rational_generators.py` — rational scalar equations, generalised Lambert-W-type inverse laws and rational mode equations.
7. `07_inverse_spherical_bessel.py` — the paper's branch-specific construction and independent full-system validation.
8. `08_modular_lambda.py` — the paper's Hamiltonian, theta-constant modes, parameter constraints, Darboux–Halphen map and validation.

Each notebook defines its own functions and runs independently.

## Read the executed notebooks

- [General Hamiltonian and reconstruction](exports/01_general_hamiltonian.html)
- [Inverse construction and equivalent forms](exports/02_inverse_construction.html)
- [Total modal phase and correction](exports/03_total_phase.html)
- [Algebraic examples and incomplete beta inversion](exports/04_algebraic_examples.html)
- [Log-augmented polynomial systems](exports/05_log_augmented.html)
- [Rational generators and generalised inverse laws](exports/06_rational_generators.html)
- [Inverse spherical Bessel construction](exports/07_inverse_spherical_bessel.html)
- [Modular lambda and Darboux–Halphen](exports/08_modular_lambda.html)

The sources are authoritative; HTML files are executed walkthroughs. Run and export them using the commands in the [root README](../README.md#computational-companion). The [mathematical tests](../tests/README.md) import the notebook-defined functions through Marimo's top-level function support.
