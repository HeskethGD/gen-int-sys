import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")
with app.setup:
    import marimo as mo
    import sympy as sp
    import mpmath as mp


@app.function
def rational_integral(f,rho):
    """Compute the separated integral for explicit rational generating data."""
    f=sp.cancel(f)
    if f==0 or not f.is_rational_function(rho):
        raise ValueError('Use a nonzero rational generating function.')
    partial_fractions=sp.apart(1/f,rho)
    primitive=sp.integrate(partial_fractions,rho)
    residual=sp.simplify(sp.diff(primitive,rho)-1/f)
    return dict(f=f,partial_fractions=partial_fractions,primitive=primitive,residual=residual)


@app.function
def inverse_law_derivative(polynomial,principal_part,factors,rho):
    """Differentiate a local log of exp(P+R)*product(rho-a)^nu=exp(z)."""
    denominator=sp.diff(polynomial+principal_part,rho)+sum(nu/(rho-a) for a,nu in factors)
    if sp.simplify(denominator)==0:raise ValueError('This relation has no regular local inverse.')
    return sp.cancel(1/denominator)


@app.function
def rational_mode_example():
    """Clear denominators only after deriving and restricting the Hamiltonian."""
    u,v,x=sp.symbols('u v x',nonzero=True)
    F=(1+x)/(1+2*x);P=sp.S.One;Q=sp.sqrt(F**2+4*x)
    H=(u+v-Q).subs(x,u*v)
    original=(sp.diff(H,v),-sp.diff(H,u))
    root=Q.subs(x,u*v)
    reduced=tuple(sp.cancel(a.subs(root,u+v)) for a in original)
    multiplier=sp.lcm([sp.denom(a) for a in reduced])
    cleared=tuple(sp.cancel(multiplier*a) for a in reduced)
    return dict(H=H,variables=(u,v),F=F,Q=Q,reduced=reduced,multiplier=multiplier,cleared=cleared)


@app.cell
def _():
    mo.md(r"""
    # 6. Rational generators and generalised Lambert-W-type inverse laws

    On a nonconstant regular branch, rho'=A(rho)/B(rho) implies
    dz/drho=B/A. Partial fractions integrate to a polynomial, rational principal
    parts and logarithms. Exponentiation therefore gives
    \[
    e^{\mathcal P(\rho)+\mathcal R(\rho)}
       \prod_\ell(\rho-a_\ell)^{\nu_\ell}=e^z.
    \]
    This is an inverse-law family, not a claim that every rational ODE is solved
    by the ordinary Lambert W function. Initial data fix integration constants
    and the local logarithmic/inverse branches. Zeros of f give separate constant
    solutions and poles delimit the domains used here.
    """)
    return


@app.cell
def _():
    rho=sp.Symbol('rho')
    rational_cases=[rho/(rho+1),(rho-1)**3*(rho+2)/(rho*rho+1),1/(1+rho*rho)]
    rational_rows=[]
    for _f in rational_cases:
        _data=rational_integral(_f,rho)
        assert _data['residual']==0
        rational_rows.append(dict(f=str(_f),separated_integral=str(_data['primitive'])))
    mo.ui.table(rational_rows,selection=None)
    return (rho,)


@app.cell
def _(rho):
    rational_from_law=inverse_law_derivative(rho**3/3+rho,1/(rho-1),[(0,2),(-2,-3)],rho)
    assert rational_from_law.is_rational_function(rho)
    # The rational quadrature of any explicit rational G is checked independently.
    quadrature=rational_integral((rho-1)**3*(rho+2)/(rho*rho+1),rho)
    G=(rho+3)/(rho-2)
    primitive=sp.integrate(sp.apart(G/quadrature['f'],rho),rho)
    assert sp.simplify(sp.diff(primitive,rho)*quadrature['f']-G)==0
    mo.md(rf"""
    ## Converse and reconstruction quadratures
    A sample inverse law with cubic polynomial exponent, a rational principal
    part, and factors of powers 2 and -3 gives
    \[
    \rho'={sp.latex(rational_from_law)}.
    \]
    Differentiating a local logarithm produces this rational function directly.
    For any rational G, integrating G(rho(z)) reduces to integrating G/f in rho;
    the code checks an explicit repeated-pole case by differentiation.
    """)
    return


@app.cell
def _():
    with mp.workdps(50):
        ordinary_errors=[]
        for _z in [mp.mpf('-0.3'),mp.mpf('0.2'),mp.mpf('0.7')]:
            _rho=lambda t:mp.lambertw(mp.exp(t))
            _r=_rho(_z)
            ordinary_errors.append(abs(mp.diff(_rho,_z)-_r/(1+_r)))
        assert max(ordinary_errors)<mp.mpf('1e-40')
    mo.md(r"""
    ## The ordinary Lambert function is one subcase
    \[
    \rho e^\rho=e^z,\qquad \rho=W(e^z),\qquad \rho'=\rho/(1+\rho).
    \]
    A 50-digit numerical derivative check verifies this subcase at three points.
    General polynomial exponents and rational principal parts need not reduce
    to ordinary W. The inverse spherical Bessel construction in the next
    notebook is another type of inverse altogether.
    """)
    return


@app.cell
def _():
    rational_modes=rational_mode_example()
    assert all(a.is_rational_function(*rational_modes['variables']) for a in rational_modes['reduced'])
    assert all(sp.denom(a)==1 for a in rational_modes['cleared'])
    mo.md(rf"""
    ## Rational mode equations on the selected level
    With one pair, F=(1+x)/(1+2x) and P=1, the canonical Hamiltonian contains
    a square root. After differentiation, the H=0 identity Q=u+v removes it
    from the mode equations. A common polynomial denominator is
    \[
    {sp.latex(rational_modes['multiplier'])}.
    \]
    Multiplying by it clears both reduced equations. On H=0 this corresponds
    to multiplication of H and a change of independent variable; it is not
    a claim that the square-root Hamiltonian is polynomial off that level.
    Zeros of the multiplier are excluded from this equivalence.
    """)
    return


if __name__=='__main__':
    app.run()
