import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")
with app.setup:
    import marimo as mo
    import sympy as sp
    import mpmath as mp
    import numpy as np
    from scipy.integrate import solve_ivp
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


@app.function
def theta_constants(tau):
    """Upper-half-plane theta series in tau, avoiding q^(1/4) phase resets."""
    tau=mp.mpc(tau)
    if mp.im(tau)<=0:raise ValueError('The theta series requires Im(tau)>0.')
    terms=max(12,int(mp.sqrt(-mp.log(mp.eps)/(mp.pi*mp.im(tau))))+4)
    theta2=2*mp.fsum(mp.exp(mp.j*mp.pi*tau*(n+mp.mpf('0.5'))**2) for n in range(terms))
    theta3=1+2*mp.fsum(mp.exp(mp.j*mp.pi*tau*n*n) for n in range(1,terms))
    theta4=1+2*mp.fsum((-1)**n*mp.exp(mp.j*mp.pi*tau*n*n) for n in range(1,terms))
    return theta2,theta3,theta4


@app.function
def modular_lambda(tau):
    t2,t3,_=theta_constants(tau)
    return (t2/t3)**4


@app.function
def modular_modes(tau,alpha1=1,alpha2=2):
    """Half-integer a0=1/2 solution, tau0=-i, on continuous theta branches."""
    t2,t3,t4=theta_constants(tau+mp.j)
    plus,minus=mp.exp(mp.j*mp.pi/4),mp.exp(-mp.j*mp.pi/4)
    return (plus*mp.sqrt(alpha1)*t2*t4/t3**2,
            plus*mp.sqrt(alpha2)*t2*t4/t3**2,
            minus*t2**3/(mp.sqrt(alpha1)*t3**2*t4),
            -minus*t4**3/(mp.sqrt(alpha2)*t2*t3**2))


@app.function
def modular_polynomial():
    """Original polynomial bracket and its canonical derivatives; no H=0 used."""
    u1,u2,v1,v2,a0,a2=sp.symbols('u1 u2 v1 v2 a0 a2')
    variables=(u1,u2,v1,v2)
    bracket=sp.sinh(a2)*(u1*u1*u2*u2+v1*v1*v2*v2)-2*sp.cosh(a2)*u1*u2*v1*v2-2*a0*(u1*v1-u2*v2-1)
    gradient=(sp.diff(bracket,v1),sp.diff(bracket,v2),-sp.diff(bracket,u1),-sp.diff(bracket,u2))
    return variables,(a0,a2),bracket,gradient


@app.function
def modular_full_functions(a0=0.5,a2=None):
    """Build the unreduced Hamiltonian RHS, retaining derivatives of K(m)^2."""
    a2=float(np.log(3)) if a2 is None else a2
    variables,params,bracket,gradient=modular_polynomial()
    B=sp.lambdify(variables,bracket.subs(dict(zip(params,(a0,a2)))),'numpy',cse=True)
    dB=sp.lambdify(variables,[g.subs(dict(zip(params,(a0,a2)))) for g in gradient],'numpy',cse=True)
    def hamiltonian(values):
        u1,u2,v1,v2=values;m=(1+u1*v1+u2*v2)/2
        K=complex(mp.ellipk(m))
        return 1j/np.pi*K*K*B(*values)
    def rhs(t,values):
        u1,u2,v1,v2=values;m=(1+u1*v1+u2*v2)/2
        K=complex(mp.ellipk(m));E=complex(mp.ellipe(m))
        Kprime=E/(2*m*(1-m))-K/(2*m)
        common=1j/np.pi*K*K
        factor_derivative=1j/np.pi*K*Kprime
        # dm/dxj=1/2: d[(i/pi)K^2]/dxj=(i/pi)K*Kprime.
        return common*np.asarray(dB(*values),dtype=complex)+factor_derivative*B(*values)*np.array([u1,u2,-v1,-v2])
    return rhs,hamiltonian


@app.function
def modular_validation(end=8,points=257,rtol=1e-10,atol=1e-12):
    times=np.linspace(0,end,points)
    with mp.workdps(35):
        exact=np.array([[complex(a) for a in modular_modes(float(t))] for t in times])
        lambda_values=np.array([complex(modular_lambda(float(t)+mp.j)) for t in times])
        rhs,H=modular_full_functions()
        solution=solve_ivp(rhs,(0,end),exact[0],method='DOP853',rtol=rtol,atol=atol,t_eval=times)
        if not solution.success:raise RuntimeError(solution.message)
        numeric=solution.y.T
        powers=numeric[:,:2]*numeric[:,2:]
        hvalues=np.array([H(y) for y in numeric])
        report=dict(max_mode_error=float(np.max(abs(numeric-exact))),max_H=float(max(abs(hvalues))),
            max_power_error=float(np.max(abs(powers-np.column_stack((lambda_values,lambda_values-1))))),
            max_invariant_error=float(max(abs(powers[:,0]-powers[:,1]-1))),
            initial_H=float(abs(H(exact[0]))),end=end)
    return dict(times=times,exact=exact,numeric=numeric,report=report)


@app.function
def modular_identity_checks():
    """Check theta identities, scalar equation and Darboux-Halphen convention."""
    with mp.workdps(55):
        theta_errors=[];ode_errors=[];mode_errors=[];dh_errors=[]
        for t in [mp.mpc('0','1'),mp.mpc('0.3','1'),mp.mpc('2.3','1')]:
            a,b,c=theta_constants(t);lam=modular_lambda(t)
            theta_errors.append(abs(b**4-a**4-c**4))
            lp=mp.diff(modular_lambda,t)
            ode_errors.append(abs(lp-4j/mp.pi*lam*(1-lam)*mp.ellipk(lam)**2))
            modes=modular_modes(t-mp.j)
            mode_errors.extend([abs(modes[0]*modes[2]-lam),abs(modes[1]*modes[3]-lam+1),
                                abs(modes[0]*modes[1]/(modes[2]*modes[3])-2)])
            lpp=mp.diff(modular_lambda,t,2);lppp=mp.diff(modular_lambda,t,3)
            A=lpp/lp;B=lp/lam;C=lp/(lam-1)
            dA=lppp/lp-A*A;dB=lpp/lam-B*B;dC=lpp/(lam-1)-C*C
            h1,h2,h3=-(A-B)/2,-(A-C)/2,-(A-B-C)/2
            dh=(-(dA-dB)/2,-(dA-dC)/2,-(dA-dB-dC)/2)
            expected=(h2*h3-h1*(h2+h3),h3*h1-h2*(h3+h1),h1*h2-h3*(h1+h2))
            dh_errors.extend(abs(a-b) for a,b in zip(dh,expected))
        return dict(theta_identity=float(max(theta_errors)),lambda_ode=float(max(ode_errors)),
                    mode_products=float(max(mode_errors)),darboux_halphen=float(max(dh_errors)))


@app.cell
def _():
    mo.md(r"""
    # 8. Modular lambda, theta modes and Darboux–Halphen

    This notebook implements the paper's two-mode Hamiltonian, with weights (2,2):
    \[
    H=\frac{i}{\pi}K(m)^2 B,\qquad m=(1+x_1+x_2)/2,
    \]
    where B is the quartic mixing bracket minus 2*a0*(x1-x2-1).
    The selected component has x1-x2=1 and U/V=coth(a2/2).
    Lambda satisfies
    \[
    \lambda'=\frac{4i}{\pi}\lambda(1-\lambda)K(\lambda)^2,\quad
    \rho=\frac14-\frac\lambda2,\quad x_1=\lambda,\quad x_2=\lambda-1.
    \]
    The numerical K argument always comes from the evolving mode products,
    never from the analytic lambda trajectory.
    """)
    return


@app.cell
def _():
    modular_identities=modular_identity_checks()
    assert all(value<1e-45 for value in modular_identities.values())
    mo.vstack([mo.md(r"""
    ## Continuous theta branches and explicit modes
    For a0=1/2, use the theta ratios printed in the paper, with alpha1=1,
    alpha2=2 and tau0=-i. The product alpha1*alpha2=2=coth(log(3)/2)
    selects the intended H=0 component. Theta constants are evaluated by their
    series in tau, avoiding a principal q^(1/4) phase reset during continuation.
    The table reports 55-digit checks at three upper-half-plane points.

    The Darboux–Halphen map is checked by differentiating lambda through order
    three and evaluating the paper's three equations. It is inherited from
    the scalar generating function; the canonical mode system is the construction
    being illustrated here.
    """),mo.ui.table([modular_identities],selection=None)])
    return


@app.cell
def _():
    modular_run=modular_validation()
    assert modular_run['report']['max_mode_error']<2e-7
    assert modular_run['report']['max_H']<2e-7
    mo.vstack([mo.md(r"""
    ## Independent integration of the original Hamilton equations
    DOP853 uses rtol=1e-10, atol=1e-12 on real tau in [0,8], with 257 samples.
    Both terms in dH are retained: K(m)^2 times dB and B times d[K(m)^2].
    The second term is zero only on the exact H=0 surface and is kept in the
    numerical vector field. Reference modes are continuous theta formulas.
    Reported errors and residuals are measured, not equated to solver tolerances.
    """),mo.ui.table([modular_run['report']],selection=None)])
    return (modular_run,)


@app.cell
def _(modular_run):
    fig,axes=plt.subplots(2,2,figsize=(9,6),layout='constrained')
    for _j,_ax in enumerate(axes.flat):
        _ax.plot(modular_run['times'],modular_run['exact'][:,_j].real,label='theta formula, real part')
        _ax.plot(modular_run['times'][::12],modular_run['numeric'][::12,_j].real,'o',markersize=3,label='original Hamilton equations')
        _ax.set(xlabel='tau',ylabel=('Re u1','Re u2','Re v1','Re v2')[_j]);_ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    plot_html=mo.as_html(fig)
    plt.close(fig)
    plot_html
    return


if __name__=='__main__':
    app.run()
