import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import sympy as sp


@app.function
def phase_data(x, weights, theta, F):
    """Compare velocities along x'=-beta F with the constructed Hamiltonian."""
    if not (len(x)==len(weights)==len(theta)):
        raise ValueError("One weight and phase function per modal power are required.")
    total = sum(b*t for b,t in zip(weights,theta))
    prescribed = tuple(-F*sum(b*sp.diff(t,a) for b,a in zip(weights,x)) for t in theta)
    hamiltonian = tuple(-F*sp.diff(total,a) for a in x)
    correction = tuple(sp.simplify(a-b) for a,b in zip(hamiltonian,prescribed))
    return dict(total=total,prescribed=prescribed,hamiltonian=hamiltonian,correction=correction,
                weighted_correction=sp.simplify(sum(b*c for b,c in zip(weights,correction))))


@app.function
def phase_hamiltonian_check(weights, x, F, total):
    """Differentiate the full Hamiltonian; impose H=0 and rho'=F afterwards.

    The substitution for U,V matches their values on the chosen branch.
    """
    n=len(x)
    u,v=sp.symbols(f"u1:{n+1}",nonzero=True),sp.symbols(f"v1:{n+1}",nonzero=True)
    U,V=sp.prod(a**b for a,b in zip(u,weights)),sp.prod(a**b for a,b in zip(v,weights))
    R=sp.prod(a**b for a,b in zip(x,weights))
    P=F/(2*sp.sqrt(R)*sp.sinh(total))
    Q=F/sp.tanh(total)
    mode_map=dict(zip(x,[a*b for a,b in zip(u,v)]))
    H=(P*(U+V)-Q).subs(mode_map)
    # Independent chain-rule expression, before applying level-set identities.
    residuals=[]
    for j,beta in enumerate(weights):
        raw=(U+V)*sp.diff(P,x[j]).subs(mode_map)-sp.diff(Q,x[j]).subs(mode_map)+beta*P.subs(mode_map)*V/(u[j]*v[j])
        residuals.append(sp.simplify(sp.diff(H,v[j])/u[j]-raw))
        # In power variables: P(U+V)=Q, P(U-V)=F.
        reduced=Q*sp.diff(P,x[j])/P-sp.diff(Q,x[j])+beta*(Q-F)/(2*x[j])
        expected=-F*sp.diff(total,x[j])-beta*F/(2*x[j])
        residuals.append(sp.simplify(reduced-expected))
    return tuple(residuals)


@app.cell
def _():
    mo.md(r"""
    # 3. Total modal phase and the reconstruction correction

    This notebook follows **Formulation in terms of total complex modal phase**.
    Write $u_j=\sqrt{x_j}e^{\phi_j}$ and $v_j=\sqrt{x_j}e^{-\phi_j}$ locally.
    Prescribe $\phi_j(z)=\theta_j(x(z))$, with $x_j'=-\beta_jF$ and
    $\widehat\Phi=\sum_j\beta_j\theta_j$. The Hamiltonian is
    \[
    H=(U+V)\frac{F}{2\sqrt R\sinh\widehat\Phi}
           -\frac{F}{\tanh\widehat\Phi}.
    \]
    All divisions and branch choices are local. Construct H first, differentiate,
    and only then use its chosen H=0 component.
    """)
    return


@app.cell
def _():
    x1,x2 = sp.symbols("x1 x2",positive=True)
    phase_x=(x1,x2)
    phase_theta=(x1*x2,x1**2)
    phase_weights=(1,1)
    phase_F=sp.S.One
    comparison=phase_data(phase_x,phase_weights,phase_theta,phase_F)
    assert comparison['weighted_correction']==0
    mo.md(rf"""
    ## A nonzero correction

    Take F=1, unit weights, $\theta_1=x_1x_2$ and $\theta_2=x_1^2$.
    The prescribed velocities are
    \[
    {sp.latex(sp.Matrix(comparison['prescribed']))},
    \]
    whereas the constructed Hamiltonian gives
    \[
    {sp.latex(sp.Matrix(comparison['hamiltonian']))}.
    \]
    Their difference is
    \[
    {sp.latex(sp.Matrix(comparison['correction']))}.
    \]
    Its weighted sum vanishes; individual phases need not agree.
    """)
    return comparison,phase_F,phase_theta,phase_weights,phase_x


@app.cell
def _(comparison,phase_x):
    rho,s = sp.symbols("rho s",real=True)
    # gamma=(1,-1), rho0=-3. The path -4<rho<-2 has positive x1,x2.
    path={phase_x[0]:1-s,phase_x[1]:-1-s}
    eta=tuple(sp.integrate(c.subs(path),(s,-3,rho)) for c in comparison['correction'])
    assert sp.simplify(sum(eta))==0
    assert all(sp.simplify(e.subs(rho,-3))==0 for e in eta)
    assert all(sp.simplify(sp.diff(e,rho)-c.subs({phase_x[0]:1-rho,phase_x[1]:-1-rho}))==0 for e,c in zip(eta,comparison['correction']))
    mo.md(rf"""
    ## Integrate along the conserved-power line

    Set $\gamma=(1,-1)$, $\rho_0=-3$, $x=(1-\rho,-1-\rho)$ and rho'=1.
    On a small domain around rho=-3, all required quantities are regular.
    With $\xi_j=\eta_j(\rho)$, the explicit correction is
    \[
    {sp.latex(sp.Matrix(eta))}.
    \]
    Each correction is zero initially. The formulas
    $\bar u_j=u_je^{{\eta_j}}$, $\bar v_j=v_je^{{-\eta_j}}$
    preserve each power. Since eta1+eta2=0, both weighted products are also
    preserved. The following tests substitute these modes into the original
    Hamilton equations independently.
    """)
    return (eta,)


@app.cell
def _(comparison,phase_F,phase_weights,phase_x):
    residuals=phase_hamiltonian_check(phase_weights,phase_x,phase_F,comparison['total'])
    assert all(r==0 for r in residuals)
    mo.md(r"""
    ## What the condition asserts

    The symbolic calculation verifies differentiation of the full Hamiltonian
    and the reduction to
    \[
    \bar\phi_j'=-F\partial_{x_j}\widehat\Phi.
    \]
    Prescribed velocities are $\phi_j'=-F\sum_k\beta_k\partial_{x_k}\theta_j$.
    Therefore on F!=0 they agree precisely when the paper's contracted
    antisymmetric derivative vanishes along the trajectory.

    This is a compatibility test for the chosen phase functions. It does not
    assert a unique canonical normal form. The correction preserves powers and
    weighted products, not necessarily each mode's global analytic character.
    """)
    return


if __name__ == "__main__":
    app.run()
