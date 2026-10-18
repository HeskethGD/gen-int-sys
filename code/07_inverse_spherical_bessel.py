import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")
with app.setup:
    import marimo as mo
    import sympy as sp
    import mpmath as mp
    import numpy as np
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    from scipy.special import sici
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt


@app.function
def j1_values(s):
    """j1, its first and second derivatives on the nonzero real branch used here."""
    if abs(s)<1e-4:
        # Analytic series also avoid removable cancellation at the origin.
        j=s/3-s**3/30+s**5/840-s**7/45360
        dj=1/3-s*s/10+s**4/168-s**6/6480
        ddj=-s/5+s**3/42-s**5/1080
        return j,dj,ddj
    j=np.sin(s)/s**2-np.cos(s)/s
    dj=np.sin(s)/s+2*np.cos(s)/s**2-2*np.sin(s)/s**3
    ddj=-2*dj/s-(1-2/s**2)*j
    return j,dj,ddj


@app.function
def bessel_initial_data():
    a=np.array([2.3,1.35]);b=np.array([6.4,9.7])
    x=np.array([5.8116,3.4584]);c=-sum(x)/2;gamma=x+c
    theta=a*sici(b*x)[0]
    # Choose alpha_1=alpha_2=1 exactly, preserving the previous modal powers.
    initial=np.concatenate((np.sqrt(x)*np.exp(theta),np.sqrt(x)*np.exp(-theta)))
    return dict(a=a,b=b,initial=initial,x0=x,c=c,gamma=gamma,theta0=theta,
                alpha=np.ones(2))


@app.function
def bessel_hamiltonian(values,data):
    u=np.asarray(values[:2]);v=np.asarray(values[2:]);x=u*v
    rho=-sum(x)/2;F=1/j1_values(rho)[1]
    phi=sum(data['a']*sici(data['b']*x)[0])
    P=F/(2*np.sqrt(x[0]*x[1])*np.sinh(phi));Q=F/np.tanh(phi)
    return P*(u[0]*u[1]+v[0]*v[1])-Q


@app.function
def bessel_original_rhs(z,values,data):
    """Differentiate H=P(U+V)-Q without imposing its zero level."""
    u=np.asarray(values[:2]);v=np.asarray(values[2:]);x=u*v
    rho=-sum(x)/2;_,dj,ddj=j1_values(rho)
    F=1/dj;Fx=ddj/(2*dj*dj)
    phi=sum(data['a']*sici(data['b']*x)[0])
    phix=data['a']*np.sin(data['b']*x)/x
    P=F/(2*np.sqrt(x[0]*x[1])*np.sinh(phi))
    Px=P*(Fx/F-1/(2*x)-phix/np.tanh(phi))
    Qx=Fx/np.tanh(phi)-F*phix/np.sinh(phi)**2
    U,V=np.prod(u),np.prod(v);common=(U+V)*Px-Qx
    return np.concatenate((common*u+P*V/v,-common*v-P*U/u))


@app.function
def bessel_validation(points=161,fraction=0.98,rtol=1e-11,atol=1e-13):
    data=bessel_initial_data();c=data['c'];jc,djc,_=j1_values(c)
    direction=np.sign(djc);left=c
    for distance in np.arange(0.05,20,0.05):
        right=c+direction*distance
        if j1_values(left)[1]*j1_values(right)[1]<=0:
            critical=brentq(lambda s:j1_values(s)[1],*sorted((left,right)))
            break
        left=right
    else:raise RuntimeError('No next critical point found.')
    if not 0<fraction<1:raise ValueError('Stop strictly before the inverse branch endpoint.')
    branch_end=j1_values(critical)[0]-jc
    times=np.linspace(0,fraction*branch_end,points)
    bracket=sorted((c,critical))
    rho=np.array([c if z==0 else brentq(lambda s:j1_values(s)[0]-jc-z,*bracket,xtol=1e-13,rtol=1e-13) for z in times])
    x=data['gamma'][None,:]-rho[:,None]
    theta=data['a'][None,:]*sici(data['b'][None,:]*x)[0]
    scale=np.sqrt(x/data['x0'][None,:])
    exact=np.concatenate((data['initial'][:2]*scale*np.exp(theta-data['theta0']),
                          data['initial'][2:]*scale*np.exp(-theta+data['theta0'])),axis=1)
    result=solve_ivp(lambda z,y:bessel_original_rhs(z,y,data),(0,times[-1]),data['initial'],
        t_eval=times,method='DOP853',rtol=rtol,atol=atol,max_step=0.05)
    if not result.success:raise RuntimeError(result.message)
    numeric=result.y.T
    error=np.max(np.abs(numeric-exact),axis=0)
    H=np.array([bessel_hamiltonian(y,data) for y in numeric])
    powers=numeric[:,:2]*numeric[:,2:]
    inverse_residual=max(abs(j1_values(r)[0]-jc-z) for r,z in zip(rho,times))
    report=dict(max_mode_error=float(max(error)),max_H=float(max(abs(H))),
        max_power_error=float(np.max(abs(powers-x))),inverse_residual=float(inverse_residual),
        branch_endpoint=float(branch_end),stop=float(times[-1]),critical_rho=float(critical),
        min_abs_j1_prime=float(min(abs(j1_values(r)[1]) for r in rho)),
        initial_H=float(bessel_hamiltonian(data['initial'],data)),
        alpha_product=float(np.prod(data['alpha'])))
    return dict(times=times,rho=rho,exact=exact,numeric=numeric,report=report,data=data)


@app.function
def bessel_symbolic_checks():
    s,x,a,b=sp.symbols('s x a b',nonzero=True)
    j=sp.sin(s)/s**2-sp.cos(s)/s
    formula=(s*s*sp.sin(s)+2*s*sp.cos(s)-2*sp.sin(s))/s**3
    return (sp.simplify(sp.diff(j,s)-formula),sp.simplify(sp.diff(a*sp.Si(b*x),x)-a*sp.sin(b*x)/x))


@app.cell
def _():
    assert bessel_symbolic_checks()==(0,0)
    mo.md(r"""
    # 7. Inverse spherical Bessel construction

    Choose the local inverse branch through c=-4.635:
    \[
    j_1(\rho(z))=j_1(c)+z,\qquad \rho'=1/j_1'(\rho),\qquad
    x_j=\gamma_j-\rho.
    \]
    Phases a_j*Si(b_j*x_j) have derivatives a_j*sin(b_j*x_j)/x_j.
    The symbolic checks verify both special-function derivatives. The numerical
    calculation inverts j1 inside its monotone interval, and stops before the
    next critical point, where the inverse derivative diverges.
    """)
    return


@app.cell
def _():
    mo.md(r"""
    ## Compatible initial data

    The displayed Hamiltonian uses the total phase without additive constants.
    Its selected H=0 branch requires alpha_1*alpha_2=1. Choose both constants
    equal to one and retain the modal powers x_1(0)=5.8116, x_2(0)=3.4584:
    \[
    u_j(0)=\sqrt{x_j(0)}\exp[a_j\operatorname{Si}(b_jx_j(0))],\qquad
    v_j(0)=\sqrt{x_j(0)}\exp[-a_j\operatorname{Si}(b_jx_j(0))].
    \]
    These values preserve rho(0)=-4.635 and its inverse branch. No additive
    phase constants are introduced into the Hamiltonian.
    """)
    return


@app.cell
def _():
    bessel_run=bessel_validation()
    assert bessel_run['report']['max_mode_error']<1e-7
    assert bessel_run['report']['max_H']<1e-7
    assert bessel_run['report']['inverse_residual']<1e-11
    assert abs(bessel_run['report']['initial_H'])<1e-12
    assert bessel_run['report']['alpha_product']==1
    mo.vstack([mo.md('## Independent full-system integration\n\nThe paper’s parameters and initial modes are used. DOP853: rtol=1e-11, atol=1e-13, max_step=0.05. The displayed interval is 98% of the available inverse branch; the endpoint itself is singular. Errors below are measured against inverse-function mode formulas.'),mo.ui.table([bessel_run['report']],selection=None)])
    return (bessel_run,)


@app.function
def bessel_figure(run):
    """Manuscript comparison: powers, phases, and all four positive modes."""
    fig=plt.figure(figsize=(10,8),layout='constrained')
    grid=fig.add_gridspec(2,2)
    power=fig.add_subplot(grid[0,0]);phase=fig.add_subplot(grid[0,1])
    modes=fig.add_subplot(grid[1,:])
    styles=['-','--','-.',':'];markers=['o','s','v','D']
    sample=np.linspace(0,len(run['times'])-1,20,dtype=int)
    t=run['times'];exact=run['exact'];numeric=run['numeric']
    for j in range(2):
        power.plot(t,exact[:,j]*exact[:,j+2],styles[j],color='black',label=f'x{j+1}')
        power.plot(t[sample],numeric[sample,j]*numeric[sample,j+2],markers[j],color='black',markersize=4)
        phase.plot(t,.5*np.log(exact[:,j]/exact[:,j+2]),styles[j],color='black',label=f'phase {j+1}')
        phase.plot(t[sample],.5*np.log(numeric[sample,j]/numeric[sample,j+2]),markers[j],color='black',markersize=4)
    for j,label in enumerate(['u1','u2','v1','v2']):
        modes.plot(t,exact[:,j],styles[j],color='black',label=label)
        modes.plot(t[sample],numeric[sample,j],markers[j],color='black',markersize=4)
    for ax,label in [(power,'Modal power'),(phase,'log(u/v)/2'),(modes,'Mode value (log scale)')]:
        ax.set(xlabel='z',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    modes.set_yscale('log')
    return fig


@app.cell
def _(bessel_run):
    fig=bessel_figure(bessel_run)
    plot_html=mo.as_html(fig)
    plt.close(fig)
    plot_html
    return


if __name__=='__main__':
    app.run()
