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
def polynomial_model(weights, quadratic=True):
    """Differentiate a representative constant-P polynomial Hamiltonian."""
    n=len(weights)
    u,v,x=sp.symbols(f'u1:{n+1}'),sp.symbols(f'v1:{n+1}'),sp.symbols(f'x1:{n+1}')
    Q=1+sum((j+1)*a for j,a in enumerate(x))+(3*x[0]**2 if quadratic else 0)
    mapping=dict(zip(x,[a*b for a,b in zip(u,v)]))
    U,V=sp.prod(a**b for a,b in zip(u,weights)),sp.prod(a**b for a,b in zip(v,weights))
    H=U+V-Q.subs(mapping)
    residuals=[]
    for j,beta in enumerate(weights):
        expected=-sp.diff(Q,x[j]).subs(mapping)*u[j]+beta*V/v[j]
        residuals.append(sp.simplify(sp.diff(H,v[j])-expected))
        power=v[j]*sp.diff(H,v[j])-u[j]*sp.diff(H,u[j])
        residuals.append(sp.simplify(power+beta*(U-V)))
    rho=sp.Symbol('rho')
    gamma=list(range(n-1))+[-sum(range(n-1))]
    line=dict(zip(x,[g-b*rho for g,b in zip(gamma,weights)]))
    scalar=sp.expand(Q.subs(line)**2-4*sp.prod(line[a]**b for a,b in zip(x,weights)))
    return dict(H=H,Q=Q,scalar=scalar,rho=rho,residuals=tuple(residuals))


@app.function
def normalized_scalar_genus(polynomial, variable):
    """Remove square factors from y^2=polynomial before computing its genus.

    Constant square parts give a reducible/degenerate relation (return None).
    This is the scalar double cover, not the full mode curve.
    """
    p=sp.Poly(polynomial,variable)
    if p.is_zero:
        return None
    _,factors=sp.sqf_list(p)
    odd_part=sp.prod(f.as_expr() for f,multiplicity in factors if multiplicity%2)
    degree=sp.degree(odd_part,variable)
    if degree<=0:
        return None
    return int((degree-1)//2)


@app.function
def fermat_symbolic(n,m):
    """Check original Hamiltonian derivatives and the beta differential."""
    if n<=1 or int(n)!=n or int(m)!=m:
        raise ValueError('Use integer n>1 and integer m.')
    u,v,rho,s=sp.symbols('u v rho s',positive=True)
    x=u*v; D=1-4*x**n; K=u**n+v**n-1
    H=D**m*K
    # Original derivatives differ from level equations by K*dP.
    residual_u=sp.simplify(sp.diff(H,v)-n*D**m*v**(n-1)-K*sp.diff(D**m,v))
    residual_v=sp.simplify(-sp.diff(H,u)+n*D**m*u**(n-1)+K*sp.diff(D**m,u))
    product_identity=sp.expand((u**n-v**n)**2-(1-4*x**n)-K*(u**n+v**n+1))
    exponent=sp.Rational(1,2)-m
    # rho=-s^(1/n)/(n*4^(1/n)); rho'=(1-s)^(m+1/2).
    drho_ds=-s**(sp.Rational(1,n)-1)/(n**2*4**sp.Rational(1,n))
    beta_density=s**(sp.Rational(1,n)-1)*(1-s)**(exponent-1)
    beta_residual=sp.simplify(drho_ds/(1-s)**(m+sp.Rational(1,2))+beta_density/(n**2*4**sp.Rational(1,n)))
    return dict(H=H,residuals=(residual_u,residual_v,product_identity,beta_residual))


@app.function
def inverse_beta_interval(target,a,b,lo,hi):
    """Safeguarded Newton inversion on an explicit real subinterval of (0,1)."""
    lo,hi=mp.mpf(lo),mp.mpf(hi)
    beta=lambda value:mp.betainc(a,b,0,value,regularized=False)
    tol=mp.mpf('1e-38')
    if abs(target-beta(lo))<tol:return lo
    if abs(target-beta(hi))<tol:return hi
    if not beta(lo)<target<beta(hi):raise ValueError('Target outside the selected beta interval.')
    value=(lo+hi)/2
    for _ in range(200):
        residual=beta(value)-target
        if abs(residual)<tol:return value
        if residual<0:lo=value
        else:hi=value
        candidate=value-residual/(value**(a-1)*(1-value)**(b-1))
        value=candidate if lo<candidate<hi else (lo+hi)/2
    raise RuntimeError('Beta inversion did not converge.')


@app.function
def fermat_validation(n,m,points=17):
    """Compare beta-inverted mode formulas with unreduced canonical integration."""
    with mp.workdps(50):
        a,b=mp.mpf(1)/n,mp.mpf('0.5')-m
        s0,send=mp.mpf('0.64'),mp.mpf('0.32')
        beta0=mp.betainc(a,b,0,s0)
        factor=n*n*mp.root(4,n)
        end=(beta0-mp.betainc(a,b,0,send))/factor
        times=np.linspace(0,float(end),points)
        exact=[];identities=[]
        for index,z in enumerate(times):
            # Use the exact endpoint for the final beta target, avoiding float roundoff.
            zmp=end if index==points-1 else mp.mpf(float(z))
            sv=inverse_beta_interval(beta0-factor*zmp,a,b,send,s0)
            difference=mp.sqrt(1-sv)
            u=((1+difference)/2)**a;v=((1-difference)/2)**a
            exact.append([float(u),float(v)])
            identities.extend([abs(u**n+v**n-1),abs(u*v-mp.root(sv/4,n)),
                               abs(mp.betainc(a,b,0,sv)-beta0+factor*zmp)])
        def rhs(z,values):
            u,v=values;x=u*v;D=1-4*x**n;P=D**m;K=u**n+v**n-1
            Px=0 if m==0 else -4*m*n*x**(n-1)*D**(m-1)
            return [n*P*v**(n-1)+K*u*Px,-n*P*u**(n-1)-K*v*Px]
        result=solve_ivp(rhs,(0,float(end)),exact[0],t_eval=times,method='DOP853',rtol=1e-11,atol=1e-13)
        if not result.success:raise RuntimeError(result.message)
        error=float(np.max(np.abs(result.y.T-np.array(exact))))
        return dict(n=n,m=m,end=float(end),mode_error=error,identity_error=float(max(identities)))


@app.cell
def _():
    mo.md(r"""
    # 4. Algebraic examples and incomplete beta inversion

    This is a compact check of the nonlinear-optics summary and higher-degree
    examples, not a rederivation of their detailed elliptic solutions or mechanical
    projections. For P=c constant and Q quadratic, the original mode equations are
    \[
    u_j'=-Q_{x_j}u_j+c\beta_j V/v_j,\qquad
    v_j'=Q_{x_j}v_j-c\beta_j U/u_j.
    \]
    The scalar degree must be checked after restriction and cancellation.
    """)
    return


@app.cell
def _():
    table_cases=[('FWM',(1,1,1,1),True),('degenerate FWM',(1,1,2),True),
        ('three-wave mixing',(1,1,1),False),('polarization',(2,2),True),
        ('couplers',(1,1),True),('two-wave mixing',(2,1),False),
        ('lemniscatic weight',(4,),True),('Dixon weight',(3,),False)]
    table_rows=[]
    for _name,_weights,_quadratic in table_cases:
        _model=polynomial_model(_weights,_quadratic)
        assert all(r==0 for r in _model['residuals'])
        table_rows.append(dict(family=_name,weights=str(_weights),scalar_degree=int(sp.degree(_model['scalar'],_model['rho'])),
            normalized_scalar_genus=normalized_scalar_genus(_model['scalar'],_model['rho'])))
    mo.ui.table(table_rows,selection=None)
    return


@app.cell
def _():
    rho=sp.Symbol('rho')
    degree_rows=[]
    for _n in (3,4,5):
        _F=-4*sp.prod(sp.Integer(j)-rho for j in range(_n))
        degree_rows.append(dict(N=_n,degree=_n,genus=normalized_scalar_genus(_F,rho)))
    mo.vstack([mo.md(r"""
    ## Extended Euler system and genus

    For unit weights, v_j=kappa_j u_j and constant Q, nontrivial compatibility
    requires product(kappa_j)=-1 and Q=0 on H=0. Then
    \[
    u_j'=-\frac{c}{\kappa_j}\prod_{k\ne j}u_k.
    \]
    Distinct scalar roots give genus floor((N-1)/2). N=4 is still elliptic;
    N=5 is genus two. We do not reproduce the companion paper's sigma machinery.
    """),mo.ui.table(degree_rows,selection=None)])
    return


@app.cell
def _():
    for _n in (3,4,5):
        for _m in (-1,0,1):
            assert all(r==0 for r in fermat_symbolic(_n,_m)['residuals'])
    mo.md(r"""
    ## Fermat family: distinguish the full curve and scalar curve

    \[
    H=[1-4(uv)^n]^m(u^n+v^n-1),\quad \rho=-uv/n,\quad D=1-4(-n\rho)^n.
    \]
    On the Fermat component, with the prefactor nonzero and finite,
    \[
    \rho'^2=D^{2m+1},\qquad u^n=\frac{1+\rho'/D^m}{2},\quad
    v^n=\frac{1-\rho'/D^m}{2}.
    \]
    On the positive real branch with u^n>v^n, set s=4(-n rho)^n. Separation gives
    \[
    B_s(1/n,1/2-m)-B_{s_0}(1/n,1/2-m)=-n^2 4^{1/n}(z-z_0).
    \]
    Negative second beta parameter is allowed on 0<s<1. Removing the square
    factor leaves Y^2=D, of genus floor((n-1)/2). The full Fermat curve has
    genus (n-1)(n-2)/2; these are different curves.
    """)
    return


@app.cell
def _():
    fermat_reports=[fermat_validation(n,m) for n in (3,4,5) for m in (-1,0,1)]
    assert all(r['identity_error']<1e-35 and r['mode_error']<1e-9 for r in fermat_reports)
    mo.vstack([mo.md('## Independent canonical integration\n\nAll nine cases use DOP853, rtol=1e-11, atol=1e-13. The beta inverse uses 50-digit arithmetic on s in [0.32,0.64]. Reported errors are measured, not inferred from solver tolerances.'),mo.ui.table(fermat_reports,selection=None)])
    return


if __name__=='__main__':
    app.run()
