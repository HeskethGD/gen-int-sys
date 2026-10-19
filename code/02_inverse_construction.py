import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import sympy as sp


@app.function
def construct_from_scalar(weights, x, F, P):
    """Construct a local square-root Hamiltonian from a modal-power extension F."""
    weights = tuple(map(sp.sympify,weights))
    if len(weights) != len(x) or not weights or any(w.is_Integer is not True for w in weights) or sum(weights)==0:
        raise ValueError("Matching integer weights with nonzero sum are required.")
    if sp.sympify(P) == 0:
        raise ValueError("P must not vanish identically; local zeros are excluded separately.")
    n = len(x)
    u,v = sp.symbols(f"u1:{n+1}",nonzero=True),sp.symbols(f"v1:{n+1}",nonzero=True)
    R = sp.prod(a**b for a,b in zip(x,weights))
    D = sp.expand(F**2+4*P**2*R)
    Q = sp.sqrt(D)
    modes = dict(zip(x,[a*b for a,b in zip(u,v)]))
    U,V = sp.prod(a**b for a,b in zip(u,weights)),sp.prod(a**b for a,b in zip(v,weights))
    H = (P*(U+V)-Q).subs(modes)
    du,dv = tuple(sp.diff(H,a) for a in v),tuple(-sp.diff(H,a) for a in u)
    return dict(x=x,u=u,v=v,F=F,P=P,R=R,radicand=D,Q=Q,H=H,du=du,dv=dv)


@app.function
def mixing_equivalence(S, t, x, P, Q, inverse):
    """Check differentials on T=h(x), not equality of off-level dynamics."""
    h = inverse
    A = P*sp.diff(S,t).subs(t,h)
    original = P*S-Q
    K = t-h
    equivalent = A*K
    residuals = [sp.simplify((sp.diff(original,a)-sp.diff(equivalent,a)).subs(t,h))
                 for a in (t,)+tuple(x)]
    return dict(original=original,h=h,A=A,K=K,equivalent=equivalent,
                inverse_residual=sp.simplify(S.subs(t,h)-Q/P),residuals=tuple(residuals))


@app.cell
def _():
    mo.md(r"""
    # 2. Constructing the Hamiltonian from a scalar equation

    This notebook accompanies the inverse-construction section and its
    subsection on equivalent mixing functions. Work locally on a selected H=0
    component, with nonzero P and modal products. Extend the chosen scalar
    $f(\rho)$ to a function F(x) along $x_j=\gamma_j-\beta_j\rho$. Then choose
    \[
    Q=\sqrt{F^2+4P^2R},\qquad R=\prod_jx_j^{\beta_j}.
    \]
    A compatible holomorphic square-root branch is required. A nonzero radicand
    at the initial point is sufficient locally. Squaring gives rho'^2=F^2;
    initial product data select the desired sign of rho'.
    """)
    return


@app.cell
def _():
    x1,x2 = sp.symbols("x1 x2",nonzero=True)
    inverse_model = construct_from_scalar((1,2),(x1,x2),1+x1-x2,1+x1*x2)
    assert sp.simplify(inverse_model['Q']**2-4*inverse_model['P']**2*inverse_model['R']-inverse_model['F']**2)==0
    mo.md(rf"""
    ## An explicit symbolic construction

    For weights (1,2), choose $F=1+x_1-x_2$ and $P=1+x_1x_2$.
    The selected local Q is
    \[
    {sp.latex(inverse_model['Q'])}.
    \]
    On a conserved-power line with gamma sum zero this F restricts to a scalar
    affine function of rho. The symbolic scalar-identity residual is zero.
    The original Hamilton equations are obtained by differentiating the full H,
    not by substituting H=0 before differentiation.
    """)
    return (inverse_model,)


@app.cell
def _():
    z = sp.symbols("z",negative=True)
    qz = sp.sqrt(1-4*z)
    solution_u,solution_v = (qz+1)/2,(qz-1)/2
    _a,_b = sp.symbols("u v",positive=True)
    simple_H = _a+_b-sp.sqrt(1+4*_a*_b)
    substitution = {_a:solution_u,_b:solution_v}
    exact_residuals = [simple_H.subs(substitution),
        solution_u*solution_v+z,
        sp.diff(solution_u,z)-sp.diff(simple_H,_b).subs(substitution),
        sp.diff(solution_v,z)+sp.diff(simple_H,_a).subs(substitution)]
    exact_residuals = [sp.simplify(sp.factor(r)) for r in exact_residuals]
    assert all(r==0 for r in exact_residuals)
    mo.md(r"""
    ## Check the original equations independently

    A one-pair example with F=P=1 has
    \[
    H=u+v-\sqrt{1+4uv},\quad \rho=z,\quad
    u=\frac{\sqrt{1-4z}+1}{2},\quad v=\frac{\sqrt{1-4z}-1}{2}.
    \]
    On a real interval z<0, both modes and the chosen square root are nonzero.
    Direct substitution into H, uv=-rho, and both original Hamilton equations
    gives zero residuals. This checks the **unsquared** choice rho'=1.
    """)
    return


@app.cell
def _():
    _a,_b = sp.symbols("u v",nonzero=True)
    factor_P = _a*_b
    factor_K = _a+_b-_a*_b-2
    factor_H = factor_P*factor_K
    multiplication_residuals = [sp.simplify(sp.diff(factor_H,_b)-factor_P*sp.diff(factor_K,_b)-factor_K*sp.diff(factor_P,_b)),
        sp.simplify(-sp.diff(factor_H,_a)+factor_P*sp.diff(factor_K,_a)+factor_K*sp.diff(factor_P,_a))]
    assert all(r==0 for r in multiplication_residuals)
    mo.md(r"""
    ## Common factors and the independent variable

    For H=PK, the product rule gives $X_H=P X_K+K X_P$.
    On K=0 the extra term vanishes and $d\xi/dz=P$ matches the trajectories.
    The explicit check uses $P=uv$ and $K=u+v-uv-2$.
    This is a selected-level identity; off that level the extra term remains.
    """)
    return


@app.cell
def _():
    t,w = sp.symbols("T x",real=True)
    pfun,qfun = 1+w,2+w**2
    exp_equivalence = mixing_equivalence(sp.exp(t),t,(w,),pfun,qfun,sp.log(qfun/pfun))
    assert exp_equivalence['inverse_residual']==0
    assert all(r==0 for r in exp_equivalence['residuals'])
    mo.md(r"""
    ## Recognising a nonlinear function of the mixing terms

    If $\mathcal H_S=P S(T)-Q$, with $T=U+V$, let
    \[
    \widetilde Q=S^{-1}(Q/P),\quad K=T-\widetilde Q,\quad A=PS'(\widetilde Q).
    \]
    Differentiation on $T=\widetilde Q$ gives $d\mathcal H_S=A\,dK$.
    Thus K gives the same local trajectories after reparameterisation, and
    $\widetilde H=AK$ gives the same mode equations in z **on the selected level**.
    Invertibility of S and nonzero P,A are local assumptions.

    The symbolic check uses S(T)=exp(T): $\widetilde Q=\log(Q/P)$, A=Q. Derivatives with respect
    to T and x agree on $T=\widetilde Q$. This does not assert equality away from that level.
    """)
    return


if __name__ == "__main__":
    app.run()
