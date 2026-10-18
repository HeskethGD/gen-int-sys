"""Independent checks of the three self-contained theory notebooks."""
import importlib.util
from pathlib import Path
import sys

import mpmath as mp
import pytest
import sympy as sp


@pytest.fixture(scope="module")
def notebooks():
    loaded = {}
    root = Path(__file__).resolve().parents[1]
    for stem in ("01_general_hamiltonian", "02_inverse_construction", "03_total_phase"):
        spec = importlib.util.spec_from_file_location(stem, root/"code"/(stem+".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[stem] = module
        spec.loader.exec_module(module)
        loaded[stem[:2]] = module
    return loaded


@pytest.mark.parametrize("weights", [(1,), (2,), (1,2), (1,-2), (1,1,1)])
def test_arbitrary_coefficient_reduction(notebooks, weights):
    data = notebooks['01'].derive_hamiltonian(weights)
    assert data['power_residuals'] == (0,)*len(weights)
    assert data['scalar_residual'] == 0
    assert all(r == 0 for r in data['reconstruction_residuals'])
    assert all(r == 0 for r in data['invariant_residuals']+data['mutual_residuals'])
    assert sp.simplify(data['rho_dot']-data['P']*(data['U']-data['V'])) == 0


def test_normalization_and_noninteger_inputs(notebooks):
    for weights in [(1,-1), (), (sp.Rational(1,2),)]:
        with pytest.raises(ValueError):
            notebooks['01'].derive_hamiltonian(weights)


@pytest.mark.parametrize("kind", ['regular','compatible_constant'])
def test_exact_reconstruction(notebooks, kind):
    data = notebooks['01'].reconstruction_example(kind)
    assert data['scalar'] == data['reconstruction'] == data['product_residual'] == 0
    if kind == 'compatible_constant':
        assert data['stationary_rate'] == 0


def test_constant_root_does_not_imply_reconstruction(notebooks):
    data = notebooks['01'].reconstruction_example('excluded_constant')
    assert data['scalar'] == data['reconstruction'] == 0
    assert data['product_residual'].subs(data['z'],0) == 0
    assert data['product_residual'].subs(data['z'],1) != 0
    assert data['stationary_rate'] != 0


def test_constant_powers_can_have_nonconstant_modes():
    z = sp.symbols('z',real=True)
    u1,u2,v1,v2 = sp.symbols('u1 u2 v1 v2')
    # H=U+V-2*x1, weights=(1,1). Its constant-power solution is not a fixed point.
    H = u1*u2+v1*v2-2*u1*v1
    variables=(u1,u2,v1,v2)
    values=(sp.exp(-z),sp.exp(z),sp.exp(z),sp.exp(-z))
    rhs=(sp.diff(H,v1),sp.diff(H,v2),-sp.diff(H,u1),-sp.diff(H,u2))
    substitution=dict(zip(variables,values))
    assert sp.simplify(H.subs(substitution)) == 0
    assert all(sp.simplify(sp.diff(value,z)-equation.subs(substitution))==0
               for value,equation in zip(values,rhs))
    assert sp.simplify(values[0]*values[1]) == sp.simplify(values[2]*values[3]) == 1
    assert sp.diff(values[0],z) != 0


@pytest.mark.parametrize('weights', [(1,), (1,2), (1,-2)])
def test_inverse_construction_original_equations(notebooks, weights):
    n=len(weights)
    x=sp.symbols(f'x1:{n+1}',positive=True)
    F=1+sum(x)
    P=1+sp.prod(x)
    model=notebooks['02'].construct_from_scalar(weights,x,F,P)
    u,v=model['u'],model['v']
    mode_map=dict(zip(x,[a*b for a,b in zip(u,v)]))
    actual_rho_dot=-sum(b*da+a*db for a,b,da,db in zip(u,v,model['du'],model['dv']))/sum(weights)
    pm,qm=P.subs(mode_map),model['Q'].subs(mode_map)
    U=sp.prod(a**b for a,b in zip(u,weights));V=sp.prod(a**b for a,b in zip(v,weights))
    # Away from H=0 the scalar residual is generally nonzero and has this factor.
    assert sp.simplify(actual_rho_dot**2-F.subs(mode_map)**2-model['H']*(pm*(U+V)+qm)) == 0


def test_inverse_exact_solution_matches_unsquared_branch():
    z=sp.symbols('z',negative=True)
    u,v=sp.symbols('u v',positive=True)
    H=u+v-sp.sqrt(1+4*u*v)
    values={u:(sp.sqrt(1-4*z)+1)/2,v:(sp.sqrt(1-4*z)-1)/2}
    assert sp.simplify(H.subs(values)) == 0
    assert sp.simplify(-sp.diff((u*v).subs(values),z)) == 1
    assert sp.simplify(sp.factor(sp.diff(values[u],z)-sp.diff(H,v).subs(values))) == 0
    assert sp.simplify(sp.factor(sp.diff(values[v],z)+sp.diff(H,u).subs(values))) == 0


@pytest.mark.parametrize('case',['exp','mobius'])
def test_nonlinear_mixing_differentials_on_level(notebooks, case):
    t,x=sp.symbols('T x',real=True)
    P,Q=1+x,2+x*x
    y=Q/P
    if case=='exp':
        S,h=sp.exp(t),sp.log(y)
    else:
        # determinant bc-ad=5; stay away from the pole and the inverse pole.
        S,h=(2+3*t)/(1-t),(y-2)/(y+3)
    data=notebooks['02'].mixing_equivalence(S,t,(x,),P,Q,h)
    assert data['inverse_residual'] == 0
    assert all(r==0 for r in data['residuals'])
    assert sp.simplify(sp.diff(data['original'],t)-sp.diff(data['equivalent'],t)) != 0


def test_hamiltonian_multiplication_is_not_off_level_equivalence():
    u,v=sp.symbols('u v')
    P=u*v; K=u+v-u*v-2; H=P*K
    assert sp.expand(sp.diff(H,v)-P*sp.diff(K,v)-K*sp.diff(P,v)) == 0
    assert sp.expand(-sp.diff(H,u)+P*sp.diff(K,u)+K*sp.diff(P,u)) == 0
    assert (sp.diff(H,v)-P*sp.diff(K,v)).subs({u:1,v:1}) != 0


def test_phase_correction_and_zero_case(notebooks):
    x=sp.symbols('x1:4',positive=True)
    weights=(1,2,1)
    theta=(x[0]*x[1], x[2]**2, x[0]+x[1])
    data=notebooks['03'].phase_data(x,weights,theta,1+x[0])
    assert data['weighted_correction']==0
    assert any(c!=0 for c in data['correction'])
    potential=x[0]*x[1]+x[1]*x[2]**2
    exact=notebooks['03'].phase_data(x,weights,tuple(sp.diff(potential,a) for a in x),1+x[0])
    assert exact['correction']==(0,0,0)


def test_phase_hamiltonian_symbolic_derivation(notebooks):
    x=sp.symbols('x1:3',positive=True)
    total=sp.Function('Phi')(*x)
    F=sp.Function('F')(*x)
    assert all(r==0 for r in notebooks['03'].phase_hamiltonian_check((1,2),x,F,total))


def test_corrected_modes_satisfy_original_hamilton_equations(notebooks):
    # Compare analytic differentiation of the corrected modes with derivatives
    # of the unreduced Hamiltonian; no reduced RHS supplies the reference.
    x=sp.symbols('x1:3',positive=True)
    rho,s=sp.symbols('rho s',real=True)
    theta=(x[0]*x[1],x[0]**2)
    data=notebooks['03'].phase_data(x,(1,1),theta,sp.S.One)
    path={x[0]:1-s,x[1]:-1-s}
    eta=[sp.integrate(c.subs(path),(s,-3,rho)) for c in data['correction']]
    xvals=(1-rho,-1-rho)
    theta_vals=[t.subs(dict(zip(x,xvals))) for t in theta]
    analytic=[sp.sqrt(xvals[j])*sp.exp(theta_vals[j]+eta[j]) for j in range(2)]
    analytic += [sp.sqrt(xvals[j])*sp.exp(-theta_vals[j]-eta[j]) for j in range(2)]
    base=[sp.sqrt(xvals[j])*sp.exp(theta_vals[j]) for j in range(2)]
    base += [sp.sqrt(xvals[j])*sp.exp(-theta_vals[j]) for j in range(2)]
    assert all(sp.simplify((a-b).subs(rho,-3))==0 for a,b in zip(analytic,base))
    u1,u2,v1,v2=sp.symbols('u1 u2 v1 v2',positive=True)
    variables=(u1,u2,v1,v2)
    mapping={x[0]:u1*v1,x[1]:u2*v2}
    total=data['total'].subs(mapping)
    H=(u1*u2+v1*v2)/(2*sp.sqrt(u1*v1*u2*v2)*sp.sinh(total))-1/sp.tanh(total)
    original_rhs=[sp.diff(H,v1),sp.diff(H,v2),-sp.diff(H,u1),-sp.diff(H,u2)]
    reference=sp.lambdify(variables,original_rhs,'mpmath')
    prescribed_reference=sp.lambdify(rho,base,'mpmath')
    prescribed_derivative=sp.lambdify(rho,[sp.diff(a,rho) for a in base],'mpmath')
    evaluate=sp.lambdify(rho,analytic,'mpmath')
    derivative=sp.lambdify(rho,[sp.diff(a,rho) for a in analytic],'mpmath')
    with mp.workdps(90):
        for point in ['-3.1','-3','-2.9']:
            t=mp.mpf(point)
            values=evaluate(t)
            for a,b in zip(derivative(t),reference(*values)):
                assert abs(a-b)/max(mp.mpf(1),abs(a),abs(b)) < mp.mpf('1e-55')
        # At matching initial data the uncorrected derivatives disagree.
        t=mp.mpf('-3')
        assert max(abs(a-b) for a,b in zip(prescribed_derivative(t),reference(*prescribed_reference(t)))) > mp.mpf('0.1')


@pytest.mark.parametrize('number',['01','02','03'])
def test_complete_notebook_execution(notebooks, number):
    # Exercise the displayed derivations too, not only imported helper functions.
    outputs, definitions = notebooks[number].app.run()
    assert outputs
    # Marimo serializes display math as a marimo-tex element with ||[ delimiters.
    # Count the expected blocks so successful execution cannot hide raw math text.
    rendered = "".join(getattr(output, "text", "") for output in outputs)
    assert rendered.count('class="arithmatex">||[') == {"01": 18, "02": 4, "03": 6}[number]
