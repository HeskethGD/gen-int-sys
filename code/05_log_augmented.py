import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")
with app.setup:
    import marimo as mo
    import sympy as sp
    import mpmath as mp
    import numpy as np
    from scipy.integrate import solve_ivp


@app.function
def sergyeyev_checks():
    X,Y,Z,b,I,J,u,v,rho=sp.symbols('X Y Z b I J u v rho',nonzero=True)
    rhs=(-2*X**2+2*Z,-3*X*Y,-2*X*Z+2*X**3+6*b*X)
    integrals=((b+X**2-Z)/Y**2,X**2/2+Z+b*sp.log(Y**2))
    residuals=[sp.simplify(sum(sp.diff(F,a)*da for a,da in zip((X,Y,Z),rhs))) for F in integrals]
    # Transform the original X,Z equations with sqrt(Z)=(v-u)/2.
    Xuv=(u+v)/2; Zuv=(v-u)**2/4; x=u*v
    dX=rhs[0].subs({X:Xuv,Z:Zuv});dZ=rhs[2].subs({X:Xuv,Z:Zuv})
    transformed=(dX-dZ/(v-u),dX+dZ/(v-u))
    bracket=-sp.Rational(3,2)*(u*u+v*v)+x+4*J-4*b*sp.log((x+b)/I)
    g=sp.sqrt(sp.Rational(8,3)*J-sp.Rational(4,3)*x-sp.Rational(8,3)*b*sp.log((x+b)/I))
    H=(x+b)*bracket/g
    # On bracket=0, the derivative of the prefactor vanishes from X_H.
    canonical=((x+b)*sp.diff(bracket,v)/(v-u),-(x+b)*sp.diff(bracket,u)/(v-u))
    residuals.extend(sp.factor(a-c) for a,c in zip(transformed,canonical))
    dx=sp.factor(v*transformed[0]+u*transformed[1])
    residuals.append(sp.factor(dx+3*(x+b)*(u+v)))
    sum_squared=sp.Rational(8,3)*(J+x-b*sp.log((x+b)/I))
    scalar=sp.Rational(9,4)*(x+b)**2*sum_squared
    expected=6*(b-2*rho)**2*(J-2*rho-b*sp.log((b-2*rho)/I))
    residuals.append(sp.simplify(scalar.subs(x,-2*rho)-expected))
    residuals.append(sp.simplify((v-u)**2-g**2+sp.Rational(2,3)*bracket))
    # Product rule verifies the full Hamiltonian differs only by bracket*d(prefactor).
    residuals.append(sp.simplify(sp.diff(H,v)-(x+b)*sp.diff(bracket,v)/g-sp.diff((x+b)/g,v)*bracket))
    return dict(H=H,residuals=tuple(residuals),scalar=expected)


@app.function
def lotka_volterra_checks():
    u,v,alpha,beta,K=sp.symbols('u v alpha beta K',nonzero=True)
    C=alpha*sp.log(-v)+beta*sp.log(u)+u+v
    H=u*v*(C-K);du=alpha*u+u*v;dv=-beta*v-u*v
    residuals=[sp.simplify(sp.diff(C,u)*du+sp.diff(C,v)*dv),
        sp.simplify(sp.diff(H,v)-du-H/v),sp.simplify(-sp.diff(H,u)-dv+H/u)]
    rho=sp.Symbol('rho',positive=True)
    P=-rho;Q=(K-alpha*sp.log(rho))*(-rho)
    expected=4*rho**3+rho**2*(K-alpha*sp.log(rho))**2
    residuals.append(sp.expand(Q**2-4*P**2*(-rho)-expected))
    return tuple(residuals)


@app.function
def pt_conservation_checks():
    p1,p2,t1,t2,chi,kappa,C=sp.symbols('p1 p2 t1 t2 chi kappa C',nonzero=True)
    variables=(p1,p2,t1,t2)
    rhs=(sp.I*chi*p1*p1*t1+kappa*p1+sp.I*p2,
         sp.I*chi*p2*p2*t2-kappa*p2+sp.I*p1,
         -sp.I*chi*p1*t1*t1+kappa*t1-sp.I*t2,
         -sp.I*chi*p2*t2*t2-kappa*t2-sp.I*t1)
    S0=p1*t1+p2*t2;S1=p1*t2+p2*t1;S2=sp.I*(p1*t2-p2*t1)
    C2=(chi*S1-2)**2+(chi*S2)**2
    logarithmic=S0-2*sp.I*kappa/chi*sp.log(sp.I*C/(2*(chi*p1*t2-1)))
    derivative=lambda expr:sp.simplify(sum(sp.diff(expr,a)*da for a,da in zip(variables,rhs)))
    residuals=(derivative(C2),derivative(logarithmic))
    # Algebraic Stokes derivatives expose the real principal-arcsin restriction.
    S3=sp.Symbol('S3',real=True)
    A,B=sp.symbols('A B',real=True)
    # S0'=2*kappa*S3, (chi*S1-2)'=chi*S3*(chi*S2).
    # sqrt(1-A^2/C^2)=abs(B/C) on a real level A^2+B^2=C^2.
    principal_rate=2*kappa*S3*(1+sp.sign(B/C))
    return dict(residuals=residuals,principal_arcsin_rate=principal_rate)


@app.function
def pt_scalar_check():
    rho,C,J,chi,kappa=sp.symbols('rho C J chi kappa',nonzero=True)
    gamma=C**2/8-sp.Rational(1,2);x1=gamma-rho;x2=-gamma-rho
    D=C**2+8*rho+4
    B=J*chi+2*sp.I*kappa*sp.log(-4*sp.I*C/D)
    P=sp.sqrt(2)*sp.sqrt(D)/4;Q=-sp.I*D*B/8
    expected=-D**2*B**2/64+(C**2-8*rho-4)*(C**2+8*rho-4)*D/128
    return sp.simplify(Q**2-4*P**2*x1*x2-expected)


@app.function
def two_mode_log_checks():
    u1,u2,v1,v2,a0,a1,a2=sp.symbols('u1 u2 v1 v2 a0 a1 a2',nonzero=True)
    L=a1*u1*v1+a2*u2*v2;T=u1*u2+v1*v2
    H=L*(sp.log(a0*L)-T)
    variables=(u1,u2,v1,v2)
    poly=(a1*u1-L*v2,a2*u2-L*v1,L*u2-a1*v1,L*u1-a2*v2)
    original=(sp.diff(H,v1),sp.diff(H,v2),-sp.diff(H,u1),-sp.diff(H,u2))
    corrections=(a1*u1,a2*u2,-a1*v1,-a2*v2)
    residuals=[sp.simplify(a-b-c*H/L) for a,b,c in zip(original,poly,corrections)]
    dx1=v1*poly[0]+u1*poly[2];dx2=v2*poly[1]+u2*poly[3]
    residuals.append(sp.expand(dx1-dx2))
    rho,gamma,w=sp.symbols('rho gamma w',nonzero=True)
    R=a1*(gamma-rho)+a2*(-gamma-rho)
    # The reciprocal scalar substitution uses log(w)=-log(a0*R) locally.
    logw=sp.Symbol('logw')
    wprime_squared=sp.cancel((a1+a2)**2/(a0**2*R**2)*(logw**2-4*(gamma-rho)*(-gamma-rho)))
    rhoval=sp.solve(sp.Eq(w,1/(a0*R)),rho)[0]
    expected=4/a0**2*(2*a0*a1*gamma*w-1)*(2*a0*a2*gamma*w+1)+(a1+a2)**2*w*w*logw**2
    residuals.append(sp.factor(wprime_squared.subs(rho,rhoval)-expected))
    return tuple(residuals)


@app.function
def three_mode_log_checks():
    u=sp.symbols('u1:4',nonzero=True);v=sp.symbols('v1:4',nonzero=True)
    a0,a2=sp.symbols('a0 a2',nonzero=True)
    x=[a*b for a,b in zip(u,v)];S=sum(x)
    U,V=sp.prod(u),sp.prod(v)
    poly_u=[(S-2*x[j])*sp.prod(v[k] for k in range(3) if k!=j)-a0*u[j]/3 for j in range(3)]
    poly_v=[-(S-2*x[j])*sp.prod(u[k] for k in range(3) if k!=j)+a0*v[j]/3 for j in range(3)]
    dx=[v[j]*poly_u[j]+u[j]*poly_v[j] for j in range(3)];dS=sp.expand(sum(dx))
    mapped_u=[S*a for a in u];mapped_v=[S*a for a in v]
    original_rho=-S**3/3
    original_U=S**3*U;original_V=S**3*V
    residuals=[]
    for j in range(3):
        # Original on-level logarithm = -(U_original+V_original)/(a0*rho).
        log_on_level=-(original_U+original_V)/(a0*original_rho)
        original_du=sp.prod(mapped_v[k] for k in range(3) if k!=j)-a0*(log_on_level+1)*mapped_u[j]/3
        original_dv=-sp.prod(mapped_u[k] for k in range(3) if k!=j)+a0*(log_on_level+1)*mapped_v[j]/3
        residuals.extend([sp.factor(dS*u[j]+S*poly_u[j]-original_du),
                          sp.factor(dS*v[j]+S*poly_v[j]-original_dv)])
    residuals.append(sp.factor(-S*S*dS-(original_U-original_V)))
    for j in (1,2):
        invariant=(x[0]-x[j])*S**2
        residuals.append(sp.expand(sum(sp.diff(invariant,u[k])*poly_u[k]+sp.diff(invariant,v[k])*poly_v[k] for k in range(3))))
    return tuple(residuals)


@app.function
def lambert_log_validation():
    """Use a=-1 and the real principal W branch, away from the singular set."""
    with mp.workdps(60):
        rows=[]
        for z in [mp.mpf('0.3'),mp.mpf('0.7'),mp.mpf('1.1')]:
            rho=lambda t:mp.lambertw(mp.exp(-t*t))
            r=rho(z);dr=mp.diff(rho,z)
            implicit=-r-mp.log(r)-z*z
            residual=dr*dr-4*r*r/(-r-1)**2*(-r-mp.log(r))
            rows.append(max(abs(implicit),abs(residual)))
        return float(max(rows))


@app.function
def lambert_mode_checks():
    """Check the full Hamiltonian first, then impose its unsquared zero level."""
    u,v,a,K=sp.symbols('u v a K',nonzero=True)
    h=sp.Symbol('h',positive=True)
    x=u*v;rho=-x;total=u+v
    P=sp.sqrt(a)*x/(1+a*x)
    H=P*(total-2*sp.sqrt((K-sp.log(-x))/a))
    rhs=(u*(2+a*v*total)/(sp.sqrt(a)*(1+a*x)*total),
         -v*(2+a*u*total)/(sp.sqrt(a)*(1+a*x)*total))
    invariant=a*total**2/4+sp.log(-x)
    drho=sp.factor(-v*rhs[0]-u*rhs[1])
    # The positive dummy h fixes the algebraic square-root branch, which is
    # then matched to (u+v)/2; no principal-root assertion is made globally.
    full=(sp.diff(H,v),-sp.diff(H,u))
    restricted=tuple(sp.simplify(f.subs(sp.log(-x),K-a*h*h)
                                .subs(h,total/2)) for f in full)
    residuals=[sp.simplify(f-r) for f,r in zip(restricted,rhs)]
    residuals.extend([
        sp.cancel(rhs[0]-rhs[1]-2/sp.sqrt(a)),
        sp.cancel(sp.diff(invariant,u)*rhs[0]+sp.diff(invariant,v)*rhs[1]),
        sp.simplify(drho-sp.sqrt(a)*rho*(u-v)/(a*rho-1)),
        sp.simplify(drho**2-4*rho**2/(a*rho-1)**2*
                    (a*rho-sp.log(rho)+invariant))])
    denominator=(1+a*x)*total
    polynomial=(u*(2+a*v*total)/sp.sqrt(a),
                -v*(2+a*u*total)/sp.sqrt(a))
    residuals.extend(sp.cancel(denominator*r-p) for r,p in zip(rhs,polynomial))
    residuals.append(sp.simplify(
        sum(sp.diff(invariant,y)*r for y,r in zip((u,v),polynomial))))
    return dict(H=H,rhs=rhs,invariant=invariant,drho=drho,
                denominator=denominator,polynomial=polynomial,
                residuals=tuple(residuals))


@app.function
def lambert_mode_pair(t,a=-1,level=0,branch=0,root_sign=1):
    """Exact modes with t=z-z0; roots must be continued from the initial data."""
    t,a,level=mp.mpc(t),mp.mpc(a),mp.mpc(level)
    if a == 0 or root_sign not in (-1,1):
        raise ValueError("Use nonzero a and a square-root sign of +1 or -1.")
    rho=-mp.lambertw(-a*mp.exp(level-t*t),branch)/a
    common=root_sign*mp.sqrt(t*t/a-rho)
    difference=t/mp.sqrt(a)
    return common+difference,common-difference


@app.function
def lambert_mode_derivative_validation():
    """High-precision equation checks on two a signs, two W branches and levels."""
    with mp.workdps(55):
        errors=[]
        for a,branch in [(-1,0),(mp.mpf('0.25'),0),(mp.mpf('0.25'),-1)]:
            for level in [mp.mpf('0'),mp.mpf('0.2')]:
                for root_sign in (-1,1):
                    for t in [mp.mpf('0.1'),mp.mpf('0.3')]:
                        u,v=lambert_mode_pair(t,a,level,branch,root_sign)
                        total=u+v;rho=-u*v
                        denominator=mp.sqrt(a)*(1+a*u*v)*total
                        rhs=(u*(2+a*v*total)/denominator,
                             -v*(2+a*u*total)/denominator)
                        for j in range(2):
                            derivative=mp.diff(
                                lambda z:lambert_mode_pair(z,a,level,branch,root_sign)[j],t)
                            errors.append(abs(derivative-rhs[j])/max(1,abs(rhs[j])))
                        errors.extend([
                            abs(a*total**2/4+mp.log(rho)-level),
                            abs(a*rho-mp.log(rho)-(t*t-level)),
                            abs(total**2-4*(level-mp.log(rho))/a)])
        return float(max(errors))


@app.function
def lambert_mode_rhs(t,modes,a):
    u,v=modes;total=u+v
    denominator=np.sqrt(complex(a))*(1+a*u*v)*total
    return np.array([u*(2+a*v*total)/denominator,
                     -v*(2+a*u*total)/denominator])


@app.function
def lambert_mode_full_rhs(t,modes,a,level=0,root_sign=1,root_anchor=None):
    """Unreduced H_K gradient, retaining terms that vanish only on H_K=0."""
    u,v=modes;x=u*v;total=u+v
    root=root_sign*np.sqrt((level-np.log(-x))/a+0j)
    if root_anchor is not None and abs(-root-root_anchor)<abs(root-root_anchor):
        root=-root
    P=np.sqrt(complex(a))*x/(1+a*x)
    dP=np.sqrt(complex(a))/(1+a*x)**2
    bracket=total-2*root
    return np.array([dP*u*bracket+P*(1+u/(a*x*root)),
                     -dP*v*bracket-P*(1+v/(a*x*root))])


@app.function
def lambert_mode_validation(a=-1,level=0,branch=0,points=41):
    """Compare exact modes with rational, full Hamiltonian and cubic flows."""
    times=np.linspace(.1,.4,points)
    with mp.workdps(45):
        exact=np.array([[complex(x) for x in lambert_mode_pair(str(t),a,level,branch)]
                        for t in times]).T
    initial=exact[:,0]
    root_anchor=initial.sum()/2
    options=dict(t_eval=times,method='DOP853',rtol=2e-12,atol=2e-13)
    rational=solve_ivp(lambert_mode_rhs,(times[0],times[-1]),initial,
                       args=(a,),**options)
    full=solve_ivp(lambert_mode_full_rhs,(times[0],times[-1]),initial,
                   args=(a,level,1,root_anchor),**options)
    if not rational.success or not full.success:
        raise RuntimeError((rational.message,full.message))
    scale=max(1,float(np.max(abs(exact))))
    u,v=full.y
    invariant=a*(u+v)**2/4+np.log(-u*v)
    root=np.sqrt((level-np.log(-u*v))/a+0j)
    root=np.where(abs(-root-root_anchor)<abs(root-root_anchor),-root,root)
    H=np.sqrt(complex(a))*u*v/(1+a*u*v)*(u+v-2*root)
    report=dict(rational_mode_error=float(np.max(abs(rational.y-exact))/scale),
                full_mode_error=float(np.max(abs(full.y-exact))/scale),
                invariant_error=float(np.max(abs(invariant-level))),
                hamiltonian_error=float(np.max(abs(H))),
                difference_error=float(np.max(abs(u-v-2*times/np.sqrt(complex(a))))))
    # Verify the polynomial clock on its own short complex-time path,
    # reconstructing t rather than identifying real clock time with real t.
    clock0=(1+a*initial.prod())*initial.sum()
    clock_end=.08/clock0
    def cubic_path(s,state):
        u,v,t=state;total=u+v
        return clock_end*np.array([
            u*(2+a*v*total)/np.sqrt(complex(a)),
            -v*(2+a*u*total)/np.sqrt(complex(a)),
            (1+a*u*v)*total])
    cubic=solve_ivp(cubic_path,(0,1),np.array([*initial,complex(times[0])]),
                    t_eval=np.linspace(0,1,25),method='DOP853',rtol=2e-12,atol=2e-13)
    if not cubic.success:
        raise RuntimeError(cubic.message)
    with mp.workdps(45):
        expected=[]
        previous_common=initial.sum()/2
        for t in cubic.y[2]:
            pair=[complex(x) for x in lambert_mode_pair(complex(t),a,level,branch)]
            common=sum(pair)/2
            if abs(-common-previous_common)<abs(common-previous_common):
                difference=(pair[0]-pair[1])/2
                pair=[-common+difference,-common-difference]
            previous_common=sum(pair)/2
            expected.append(pair)
    expected=np.array(expected).T
    report['cubic_mode_error']=float(np.max(abs(cubic.y[:2]-expected))/scale)
    report['min_abs_denominator']=float(np.min(abs((1+a*u*v)*(u+v))))
    return report


@app.cell
def _():
    mo.md(r"""
    # 5. Log-augmented polynomial systems

    This notebook checks every example developed in the corresponding paper
    section. Polynomial mode equations need not imply an algebraic scalar curve:
    logarithmic first integrals can survive the reduction. Variables depend on z;
    parameters and fixed integral values are constant. All logarithms and roots
    are local branches, away from their singularities.

    ## First-power logarithms: Sergyeyev
    The polynomial three-variable system has first integrals
    \[
    I=(b+X^2-Z)/Y^2,\qquad J=X^2/2+Z+b\log(Y^2).
    \]
    With u=X-sqrt(Z), v=X+sqrt(Z), the prefactor is an algebraic function of uv
    on the conserved surface. Differentiation of the full Hamiltonian precedes
    restriction to H=0; the prefactor-derivative terms then vanish.
    """)
    return


@app.cell
def _():
    sergyeyev=sergyeyev_checks()
    assert all(r==0 for r in sergyeyev['residuals'])
    mo.md(rf"""
    Both first integrals, the original-time coordinate transformation and
    the scalar equation are checked symbolically:
    \[
    {sp.latex(sergyeyev['scalar'])}.
    \]
    The Hamiltonian representation is included for completeness, not as a claim
    that it simplifies the original formulation.
    """)
    return


@app.cell
def _():
    lambert_error=lambert_log_validation()
    assert lambert_error<1e-50
    mo.md(rf"""
    ## A separate explicit Lambert-W solution
    \[
    \rho=-\frac1aW_k(-a e^{{-(z-z_0)^2}}),\qquad
    \rho'^2=\frac{{4\rho^2}}{{(a\rho-1)^2}}(a\rho-\log\rho).
    \]
    The chosen log branch satisfies a*rho-log(rho)=(z-z0)^2. A 60-digit check
    on the real a=-1 branch gives maximum implicit/equation residual {lambert_error:.2e}.
    This example is independent of Sergyeyev's system.
    """)
    return


@app.cell
def _():
    lambert_modes=lambert_mode_checks()
    assert all(r==0 for r in lambert_modes['residuals'])
    mo.md(r"""
    ## A one-mode Hamiltonian for the squared-time Lambert solution

    Use the paper's parameter a, with t=z-z0 and rho=-uv. For a fixed conserved
    value K, take
    \[
    H_K=\frac{\sqrt a\,uv}{1+a uv}
       \left[u+v-2\sqrt{\frac{K-\log(-uv)}a}\right].
    \]
    Differentiating the full Hamiltonian before imposing its zero level, and
    matching the root to (u+v)/2, gives
    \[
    u'=\frac{u[2+a v(u+v)]}{\sqrt a(1+a uv)(u+v)},\qquad
    v'=-\frac{v[2+a u(u+v)]}{\sqrt a(1+a uv)(u+v)}.
    \]
    The symbolic checks verify both Hamilton equations on that branch, as well as
    \[
    (u-v)'=\frac2{\sqrt a},\qquad
    C=\frac a4(u+v)^2+\log(-uv),\qquad C'=0.
    \]
    With C=K, the explicit modes are
    \[
    \rho=-\frac1aW_k(-a e^{K-t^2}),\qquad
    u=\frac t{\sqrt a}+\sqrt{\frac{t^2}a-\rho},\quad
    v=-\frac t{\sqrt a}+\sqrt{\frac{t^2}a-\rho}.
    \]
    The common root may have either local sign. Initial modes determine
    z0=z*-(sqrt(a)/2)(u*-v*) and K=C(u*,v*); choose the W branch and root
    sign consistently with them. K=0 gives the paper's scalar equation.

    Work where a, uv, 1+a*uv and u+v are nonzero, with continued local
    logarithms and roots. For fixed K the rational equations agree with H_K
    on its selected zero-energy branch; they need not agree away from it.

    **All symbolic residuals vanish.**
    """)
    return


@app.cell
def _():
    lambert_mode_error=lambert_mode_derivative_validation()
    assert lambert_mode_error<1e-45
    lambert_mode_reports=[]
    for _a,_level,_branch in [(-1,0,0),(.25,.2,0),(.25,0,-1)]:
        _report=lambert_mode_validation(_a,_level,_branch)
        assert max(v for k,v in _report.items() if k!='min_abs_denominator')<1e-8
        assert _report['min_abs_denominator']>0
        lambert_mode_reports.append(dict(a=_a,K=_level,branch=_branch,**_report))
    mo.vstack([
        mo.md(f"""
        ### Independent numerical checks

        At 55-digit precision, derivatives of the explicit modes, the conserved
        level and the Lambert identity have maximum residual
        **{lambert_mode_error:.2e}**. The checks cover both signs of the common
        square root, positive and negative a, two conserved levels, and W branches
        0 and -1 on intervals where the selected logarithm agrees.

        DOP853 integration of both the rational equations and the unreduced
        Hamilton equations is compared against the explicit modes below.
        The latter retains all terms proportional to H_K. The integration uses
        rtol=2e-12 and atol=2e-13 for t in [0.1,0.4].
        """),
        mo.ui.table(lambert_mode_reports,selection=None)])
    return


@app.cell
def _():
    mo.md(r"""
    ### Cubic equations in a polynomial clock

    With D=(1+a*uv)(u+v) and dσ/dz=1/D, the rational equations become
    \[
    \frac{du}{d\sigma}=\frac{u[2+a v(u+v)]}{\sqrt a},\qquad
    \frac{dv}{d\sigma}=-\frac{v[2+a u(u+v)]}{\sqrt a},\qquad
    \frac{dz}{d\sigma}=(1+a uv)(u+v).
    \]
    These cubic mode equations preserve C and trace the same local solution curves.
    Multiplying H_K by D gives their canonical Hamiltonian on the selected zero
    level; additional product-rule terms vanish there.

    The table's cubic-mode error comes from independently integrating this
    augmented polynomial system along a short complex σ path, reconstructing t
    and comparing with the Lambert modes there. The square root is continued by
    matching neighbouring values. The exact formulas are parametric solutions
    in this clock: real σ is not identified with real z.
    """)
    return


@app.cell
def _():
    assert all(r==0 for r in lotka_volterra_checks())
    mo.md(r"""
    ## Lotka–Volterra: squared logarithms
    The general rescaled equations conserve alpha*log(-v)+beta*log(u)+u+v.
    Multiplying the fixed-level relation by uv gives a canonical Hamiltonian
    reproducing those equations on H=0. For beta=alpha and rho=-uv,
    \[
    \rho'^2=4\rho^3+\rho^2(K-\alpha\log\rho)^2.
    \]
    The conservation law, Hamiltonian product rule and scalar equation are
    verified symbolically; no global merging of logarithm branches is assumed.
    """)
    return


@app.cell
def _():
    pt=pt_conservation_checks()
    assert pt['residuals']==(0,0)
    assert pt_scalar_check()==0
    mo.md(r"""
    ## PT dimer: conserved logarithm versus principal inverse sine
    Direct differentiation of the original complexified four-mode equations gives
    C^2'=0 and J'=0 for the paper's logarithmic J, wherever the expression is defined.
    The canonical scalar equation is checked independently from P,Q and its
    conserved power line.

    On a real level, put A=chi*S1-2 and B=chi*S2. Since
    A^2+B^2=C^2, the derivative of the principal inverse-sine expression is
    \[
    J_{\rm principal}'=2\kappa S_3[1+\operatorname{sign}(B/C)].
    \]
    It is conserved on the sign branch C*chi*S2<0 (apart from special zero-rate
    points). The logarithm must also be continued along trajectories; fixing
    its principal value anew is not a global conservation prescription.
    """)
    return


@app.cell
def _():
    assert all(r==0 for r in three_mode_log_checks())
    assert all(r==0 for r in two_mode_log_checks())
    mo.md(r"""
    ## Constructed three- and two-mode examples
    For the restricted three-mode Hamiltonian, u_j=S*tilde(u_j),
    v_j=S*tilde(v_j), S=sum(tilde(u_j)*tilde(v_j)) maps the quartic polynomial
    equations to the original logarithmic canonical equations on H=0.
    The calculation checks all six components, rho=-S^3/3 and both power invariants.

    For the two-mode example, differentiating H=L[log(a0*L)-U-V] gives the
    displayed polynomial equations plus terms proportional to H/L. The power
    difference is conserved. Substituting w=1/(a0*ell(rho)) verifies the paper's
    final logarithmically deformed scalar equation, with
    ell(rho)=a1*(gamma1-rho)+a2*(gamma2-rho).

    **All symbolic residuals vanish.** The tests also check the distinction
    between a local conservation identity and a principal-branch restriction.
    """)
    return



@app.cell
def _():
    mo.md(r"""
    ## Weierstrass form under reparameterisation

    This supplementary derivation concerns the logarithmically deformed scalar
    equation, not the common-factor Hamiltonian lemma used in the main construction.
    Let the coefficients be constant and write
    \[
    s'^2=W(s)+c_5(s+c_6)^2\log^2\left(\frac{4(s+c_6)}{c_0}\right),
    \qquad W(s)=4s^3-g_2s-g_3,\qquad c_0\ne0.
    \]
    All expressions below are local, on consistent logarithmic and square-root
    branches, avoiding zeros of displayed denominators. Choose a solution of the
    ordinary Weierstrass equation
    \[
    \left(\frac{dw}{dh}\right)^2=W(w),\qquad
    w(h)=\wp(h-h_*;g_2,g_3),
    \]
    with a shift h_*. For singular cubic invariants, use the corresponding
    degenerate Weierstrass solution on a regular local branch. Seek
    \[
    s(z)=w(h(z)).
    \]
    Substitution gives
    \[
    \left(\frac{dw}{dh}\right)^2h'^2
      =\left(\frac{dw}{dh}\right)^2+
        c_5(w+c_6)^2\log^2\left(\frac{4(w+c_6)}{c_0}\right).
    \]
    Define the double-logarithmic function
    \[
    L(h)=\log\left[\log\left(\frac{4(w(h)+c_6)}{c_0}\right)\right].
    \]
    On a branch where this is defined,
    \[
    L'(h)=\frac{dw/dh}{(w+c_6)\log(4(w+c_6)/c_0)},\qquad
    h'^2=1+\frac{c_5}{L'(h)^2}.
    \]
    With h_0=h(z_0), introduce
    \[
    \mathcal Z(h)=\int_{h_0}^{h}
      \frac{L'(\tau)}{\sqrt{L'(\tau)^2+c_5}}\,d\tau.
    \]
    Choosing the square-root branch consistently with the sign of h' gives
    \[
    z-z_0=\mathcal Z(h(z)).
    \]
    Wherever this map is locally invertible,
    \[
    s(z)=\wp\left(\mathcal Z^{-1}(z-z_0)-h_*;g_2,g_3\right).
    \]
    Thus the Weierstrass representation remains available, but it requires
    inversion of a generally nonelliptic integral. It does **not** imply that
    s is an elliptic function of the original independent variable z.

    ### A dominant-logarithm approximation

    If
    \[
    |c_5|\gg|L'(h)|^2
    \]
    throughout the part of the trajectory considered, expansion on a chosen branch
    of the square root gives the leading approximation
    \[
    \mathcal Z(h)\sim\frac{L(h)-L(h_0)}{\sqrt{c_5}}.
    \]
    Consequently, on a regular inverse branch of L,
    \[
    L(h(z))\sim L(h_0)+\sqrt{c_5}(z-z_0),\qquad
    h(z)\sim L^{-1}\left(L(h_0)+\sqrt{c_5}(z-z_0)\right).
    \]
    This is a local leading-order approximation under the stated dominance
    assumption, not a global error or continuation estimate.

    For c_5=0, the reparameterisation integrand is locally plus or minus one,
    so h differs from z by a sign and a constant and the ordinary Weierstrass
    solution is recovered.
    """)
    return


if __name__=='__main__':
    app.run()
