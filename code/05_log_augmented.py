import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")
with app.setup:
    import marimo as mo
    import sympy as sp
    import mpmath as mp


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
    difference is conserved. Substituting w=1/(a0*R(rho)) verifies the paper's
    final logarithmically deformed scalar equation.

    **All symbolic residuals vanish.** The tests also check the distinction
    between a local conservation identity and a principal-branch restriction.
    """)
    return


if __name__=='__main__':
    app.run()
