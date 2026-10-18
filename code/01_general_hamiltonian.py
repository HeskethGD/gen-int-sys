import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import sympy as sp


@app.function
def derive_hamiltonian(weights):
    """Differentiate the unreduced Hamiltonian for explicit integer weights.

    P and Q remain arbitrary functions. Divisions are identities on a local
    nonzero-mode, nonzero-P domain, not assertions about their singular sets.
    """
    weights = tuple(map(sp.sympify, weights))
    if not weights or any(w.is_Integer is not True for w in weights):
        raise ValueError("Provide integer weights.")
    if sum(weights) == 0:
        raise ValueError("The paper's rho normalization requires nonzero weight sum.")
    n = len(weights)
    u = sp.symbols(f"u1:{n+1}", nonzero=True)
    v = sp.symbols(f"v1:{n+1}", nonzero=True)
    x = sp.symbols(f"x1:{n+1}", nonzero=True)
    p, q = sp.Function("P")(*x), sp.Function("Q")(*x)
    power_map = dict(zip(x, [a*b for a, b in zip(u, v)]))
    pm, qm = p.subs(power_map), q.subs(power_map)
    U = sp.prod(a**w for a, w in zip(u, weights))
    V = sp.prod(a**w for a, w in zip(v, weights))
    H = pm*(U+V)-qm
    du = tuple(sp.diff(H, a) for a in v)
    dv = tuple(-sp.diff(H, a) for a in u)
    dx = tuple(sp.expand(b*da+a*db) for a,b,da,db in zip(u,v,du,dv))
    rho_dot = sp.simplify(-sum(dx)/sum(weights))
    R = sp.prod((a*b)**w for a,b,w in zip(u,v,weights))
    # A residual vanishing identically verifies the asserted relation.
    power_residuals = tuple(sp.simplify(dxj+w*rho_dot) for dxj,w in zip(dx,weights))
    scalar_residual = sp.simplify(
        rho_dot**2-(qm**2-4*pm**2*R)-H*(pm*(U+V)+qm)
    )
    reconstruction_residuals = []
    for j, w in enumerate(weights):
        pj = sp.diff(p,x[j]).subs(power_map)
        qj = sp.diff(q,x[j]).subs(power_map)
        proposed_u = qm*pj/pm-qj+w*(qm-rho_dot)/(2*u[j]*v[j])
        proposed_v = -qm*pj/pm+qj-w*(qm+rho_dot)/(2*u[j]*v[j])
        correction = H*(pj/pm+w/(2*u[j]*v[j]))
        reconstruction_residuals.extend([
            sp.simplify(du[j]/u[j]-proposed_u-correction),
            sp.simplify(dv[j]/v[j]-proposed_v+correction),
        ])
    invariants = tuple(weights[-1]*u[j]*v[j]-weights[j]*u[-1]*v[-1]
                       for j in range(n-1))
    def bracket(a,b):
        return sp.simplify(sum(sp.diff(a,uj)*sp.diff(b,vj)-sp.diff(a,vj)*sp.diff(b,uj)
                               for uj,vj in zip(u,v)))
    invariant_residuals = tuple(bracket(c,H) for c in invariants)
    mutual_residuals = tuple(bracket(a,b) for a in invariants for b in invariants)
    return dict(weights=weights,u=u,v=v,x=x,P=pm,Q=qm,U=U,V=V,H=H,
                du=du,dv=dv,dx=dx,rho_dot=rho_dot,R=R,invariants=invariants,
                power_residuals=power_residuals,scalar_residual=scalar_residual,
                reconstruction_residuals=tuple(reconstruction_residuals),
                invariant_residuals=invariant_residuals,mutual_residuals=mutual_residuals)


@app.function
def reconstruction_example(kind="regular"):
    """Exact one-pair checks, including the excluded constant-root example."""
    z = sp.symbols("z", positive=True)
    if kind == "regular":
        # H=u+v, rho=z^2. Work away from z=0.
        u,v,rho,p,q,qx = z,-z,z**2,sp.S.One,sp.S.Zero,sp.S.Zero
    elif kind == "compatible_constant":
        # H=u+v-2*sqrt(uv), branch sqrt(1)=1.
        u,v,rho,p,q,qx = sp.S.One,sp.S.One,-sp.S.One,sp.S.One,sp.Integer(2),sp.S.One
    elif kind == "excluded_constant":
        # H=u+v-2: the squared scalar root and initial products alone fail.
        u,v,rho,p,q,qx = sp.exp(z),sp.exp(-z),-sp.S.One,sp.S.One,sp.Integer(2),sp.S.Zero
    else:
        raise ValueError(kind)
    x = u*v
    scalar = sp.simplify(sp.diff(rho,z)**2-q**2+4*p**2*x)
    reconstruction = sp.simplify(sp.diff(u,z)/u-(q/(2*x)-qx+sp.diff(x,z)/(2*x)))
    return dict(z=z,u=u,v=v,rho=rho,scalar=scalar,reconstruction=reconstruction,
                product_residual=sp.simplify(u-(q+sp.diff(rho,z))/(2*p)),
                stationary_rate=sp.simplify(q/(2*x)-qx))


@app.cell
def _():
    mo.md(r"""
    # 1. The general Hamiltonian and reconstruction

    This notebook accompanies **The general Hamiltonian system** and Appendix A.
    We differentiate the unreduced Hamiltonian before restricting to $H=0$:
    \[
    H=P(x)(U+V)-Q(x),\quad x_j=u_jv_j,\quad
    U=\prod_j u_j^{\beta_j},\quad V=\prod_j v_j^{\beta_j}.
    \]
    P and Q remain arbitrary symbolic functions. The calculations below check
    finite mode counts; the paper supplies the argument for general N.

    Work locally on compatible holomorphic branches, with nonzero modes and P,
    and integer weights whose sum is nonzero. A prime differentiates in z.
    """)
    return


@app.cell
def _():
    general = derive_hamiltonian((1,2))
    mo.md(rf"""
    ## Differentiate first

    For the representative weights $(1,2)$, the unreduced Hamiltonian is
    \[
    {sp.latex(general['H'])}.
    \]
    Its first mode equation is
    \[
    u_1'={sp.latex(general['du'][0])}.
    \]
    The products satisfy $x_j'=-\beta_j P(U-V)$; hence
    $\rho'=P(U-V)$ and $x_j=\gamma_j-\beta_j\rho$.
    """)
    return (general,)


@app.cell
def _(general):
    assert all(r == 0 for r in general['power_residuals'])
    assert general['scalar_residual'] == 0
    assert all(r == 0 for r in general['reconstruction_residuals'])
    assert all(r == 0 for r in general['invariant_residuals']+general['mutual_residuals'])
    mo.md(r"""
    ## Restrict to the level only after differentiation

    The symbolic calculation verifies the **off-level** identity
    \[
    (\rho')^2-(Q^2-4P^2R)=H\,[P(U+V)+Q].
    \]
    Thus the scalar reduction follows on H=0. Likewise the reconstructed
    logarithmic mode derivatives differ from the original ones by terms
    proportional to H; the code checks those terms explicitly.

    The weighted power invariants commute with H and with each other.
    Independence is a separate regular-locus condition. In particular,
    $P(U-V)\ne0$ is sufficient; no global torus claim follows from these checks.
    **All symbolic residuals above vanish.**
    """)
    return


@app.cell
def _():
    regular_case = reconstruction_example()
    compatible_case = reconstruction_example("compatible_constant")
    excluded_case = reconstruction_example("excluded_constant")
    assert regular_case['scalar'] == regular_case['reconstruction'] == regular_case['product_residual'] == 0
    assert compatible_case['scalar'] == compatible_case['reconstruction'] == compatible_case['product_residual'] == 0
    assert excluded_case['scalar'] == excluded_case['reconstruction'] == 0
    assert excluded_case['product_residual'] != 0
    mo.md(r"""
    ## Initial compatibility and constant powers

    At the reference point impose $U(z_0)=(q(z_0)+\rho'(z_0))/(2p(z_0))$.
    The direct theorem applies where rho' is nonzero; identities continue through
    isolated turning points only where the holomorphic quantities extend regularly.

    * Regular example: $H=u+v$, $u=z$, $v=-z$, $\rho=z^2$, away from z=0.
    * Compatible constant example: $H=u+v-2\sqrt{uv}$, $u=v=1$, $\rho=-1$.
    * Excluded example: $P=1,Q=2,\rho=-1,u=e^z,v=e^{-z}$.
      The squared scalar equation and reconstruction equation hold, and U=1
      initially at z=0, but the product identity is not preserved.

    In the constant-power case the missing condition is
    \[
    \sum_j\beta_j\left(\frac{\beta_jq}{2x_j}-Q_{x_j}
      +\frac{q}{p}P_{x_j}\right)=0.
    \]
    It makes U and V constant; individual modes need not be constant.
    """)
    return


if __name__ == "__main__":
    app.run()
