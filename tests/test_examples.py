"""Paper-example verification, separate from notebook presentation."""
import importlib.util
from pathlib import Path
import sys
import mpmath as mp
import numpy as np
import pytest
import sympy as sp


@pytest.fixture(scope='module')
def examples():
    root=Path(__file__).resolve().parents[1]
    loaded={}
    for stem in ['04_algebraic_examples','05_log_augmented','06_rational_generators','07_inverse_spherical_bessel','08_modular_lambda']:
        spec=importlib.util.spec_from_file_location(stem,root/'code'/(stem+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[stem]=module;spec.loader.exec_module(module)
        loaded[stem[:2]]=module
    return loaded


@pytest.mark.parametrize('weights,quadratic',[( (1,1,1,1),True),((1,1,2),True),((1,1,1),False),((2,2),True),((1,1),True),((2,1),False),((4,),True),((3,),False)])
def test_polynomial_table_equations(examples,weights,quadratic):
    data=examples['04'].polynomial_model(weights,quadratic)
    assert all(r==0 for r in data['residuals'])
    assert sp.degree(data['scalar'],data['rho'])<=4


def test_normalization_removes_square_factors(examples):
    x=sp.Symbol('x');genus=examples['04'].normalized_scalar_genus
    assert genus((x-1)**2*(x**3+1),x)==1
    assert genus((1-4*x**5)**3,x)==2
    assert genus((1-4*x**5)**2,x) is None
    assert genus(x**7-x+1,x)==3


@pytest.mark.parametrize('n',[3,4,5])
@pytest.mark.parametrize('m',[-1,0,1])
def test_fermat_beta_derivation_and_original_integration(examples,n,m):
    assert all(r==0 for r in examples['04'].fermat_symbolic(n,m)['residuals'])
    report=examples['04'].fermat_validation(n,m,points=13)
    assert report['identity_error']<1e-35
    assert report['mode_error']<1e-9


def test_beta_inverse_stays_in_selected_interval(examples):
    with mp.workdps(50):
        a,b=mp.mpf(1)/3,-mp.mpf('0.5');target=mp.betainc(a,b,0,mp.mpf('0.5'))
        result=examples['04'].inverse_beta_interval(target,a,b,'0.3','0.7')
        assert abs(result-mp.mpf('0.5'))<mp.mpf('1e-35')
        with pytest.raises(ValueError):examples['04'].inverse_beta_interval(target,a,b,'0.1','0.2')


def test_extended_euler_proportional_mode_restriction():
    # Directly differentiate H=U+V before imposing v=kappa*u, Q=0.
    for n in (3,4,5):
        u,v=sp.symbols(f'u1:{n+1}',nonzero=True),sp.symbols(f'v1:{n+1}',nonzero=True)
        kappas=(-1,)+(1,)*(n-1);H=sp.prod(u)+sp.prod(v)
        mapping=dict(zip(v,[k*a for k,a in zip(kappas,u)]))
        assert sp.expand(H.subs(mapping))==0
        for j,k in enumerate(kappas):
            du=sp.diff(H,v[j]).subs(mapping);dv=-sp.diff(H,u[j]).subs(mapping)
            assert sp.expand(dv-k*du)==0
            assert sp.expand(du+sp.prod(u[i] for i in range(n) if i!=j)/k)==0


def test_sergyeyev_integrals_coordinates_and_scalar(examples):
    assert all(r==0 for r in examples['05'].sergyeyev_checks()['residuals'])


def test_lotka_volterra_and_two_constructed_log_systems(examples):
    for name in ['lotka_volterra_checks','two_mode_log_checks','three_mode_log_checks']:
        assert all(r==0 for r in getattr(examples['05'],name)())


def test_pt_log_integral_and_principal_sign(examples):
    data=examples['05'].pt_conservation_checks()
    assert data['residuals']==(0,0)
    assert examples['05'].pt_scalar_check()==0
    expression=data['principal_arcsin_rate']
    byname={str(a):a for a in expression.free_symbols}
    assert expression.subs({byname['B']:-2,byname['C']:1})==0
    assert expression.subs({byname['B']:2,byname['C']:1,byname['kappa']:1,byname['S3']:1})==4


def test_explicit_log_lambert_solution(examples):
    assert examples['05'].lambert_log_validation()<1e-50


def test_lambert_mode_hamiltonian_invariant_and_scalar(examples):
    assert all(r==0 for r in examples['05'].lambert_mode_checks()['residuals'])


def test_lambert_mode_explicit_derivatives(examples):
    assert examples['05'].lambert_mode_derivative_validation()<1e-45


@pytest.mark.parametrize('a,level,branch',[(-1,0,0),(-1,.2,0),(.25,0,-1),(.25,.2,0)])
def test_lambert_modes_and_polynomial_clock(examples,a,level,branch):
    report=examples['05'].lambert_mode_validation(a,level,branch)
    assert report['min_abs_denominator']>0
    assert max(value for key,value in report.items() if key!='min_abs_denominator')<1e-8


@pytest.mark.parametrize('a,branch',[(-1,0),(.25,-1)])
def test_lambert_full_hamiltonian_gradient_off_level(examples,a,branch):
    module=examples['05']
    with mp.workdps(55):
        initial=module.lambert_mode_pair('.2',a,0,branch)
        values=[initial[0]*mp.mpf('1.01'),initial[1]]
        anchor=sum(initial)/2
        def H(u,v):
            x=u*v
            root=mp.sqrt(-mp.log(-x)/a)
            if abs(-root-anchor)<abs(root-anchor):
                root=-root
            return mp.sqrt(a)*x/(1+a*x)*(u+v-2*root)
        assert abs(H(*values))>mp.mpf('1e-6')
        expected=np.array([complex(mp.diff(lambda v:H(values[0],v),values[1])),
                           complex(-mp.diff(lambda u:H(u,values[1]),values[0]))])
        actual=module.lambert_mode_full_rhs(0,np.array(list(map(complex,values))),a,
                                           root_anchor=complex(anchor))
        assert np.max(abs(actual-expected))/max(1,np.max(abs(expected)))<1e-12


@pytest.mark.parametrize('f',[
    lambda r:r/(1+r),lambda r:(r-1)**3*(r+2)/(1+r*r),lambda r:1/(1+r*r)])
def test_rational_separation_and_inverse_derivative(examples,f):
    r=sp.Symbol('rho');data=examples['06'].rational_integral(f(r),r)
    assert data['residual']==0
    assert not data['primitive'].has(sp.Integral)
    recovered=1/sp.diff(data['primitive'],r)
    assert sp.simplify(recovered-f(r))==0


def test_rational_mode_denominator_clearing(examples):
    data=examples['06'].rational_mode_example()
    assert all(a.is_rational_function(*data['variables']) for a in data['reduced'])
    assert all(sp.denom(a)==1 for a in data['cleared'])
    u,v=data['variables'];multiplier=data['multiplier'];H=data['H']
    assert sp.simplify(sp.diff(multiplier*H,v)-multiplier*sp.diff(H,v)-H*sp.diff(multiplier,v))==0
    assert sp.simplify(-sp.diff(multiplier*H,u)+multiplier*sp.diff(H,u)+H*sp.diff(multiplier,u))==0


def test_bessel_full_gradient_off_level(examples):
    module=examples['07'];data=module.bessel_initial_data()
    assert module.bessel_symbolic_checks()==(0,0)
    assert abs(module.bessel_hamiltonian(data['initial'],data))<1e-12
    alpha=data['initial'][:2]/np.sqrt(data['x0'])*np.exp(-data['theta0'])
    assert np.allclose(alpha,1,rtol=0,atol=1e-14)
    assert abs(np.prod(alpha)-1)<1e-14
    assert np.allclose(data['initial'][:2]*data['initial'][2:],data['x0'],rtol=0,atol=2e-15)
    with mp.workdps(55):
        a=list(map(mp.mpf,data['a']));b=list(map(mp.mpf,data['b']))
        def H(*values):
            u=values[:2];v=values[2:];x=[u[j]*v[j] for j in range(2)];rho=-sum(x)/2
            dj=mp.sin(rho)/rho+2*mp.cos(rho)/rho**2-2*mp.sin(rho)/rho**3
            F=1/dj;phi=sum(a[j]*mp.si(b[j]*x[j]) for j in range(2))
            return F*((u[0]*u[1]+v[0]*v[1])/(2*mp.sqrt(x[0]*x[1])*mp.sinh(phi))-1/mp.tanh(phi))
        values=data['initial']*np.array([1.01,1,1,1])
        high=[mp.mpf(float(v)) for v in values]
        grad=[mp.diff(lambda t:H(*(high[:j]+[t]+high[j+1:])),high[j]) for j in range(4)]
        expected=np.array([float(grad[2]),float(grad[3]),-float(grad[0]),-float(grad[1])])
        assert np.max(abs(module.bessel_original_rhs(0,values,data)-expected))<1e-10


def test_bessel_trajectory(examples):
    report=examples['07'].bessel_validation(points=49,fraction=.98)['report']
    assert report['max_mode_error']<1e-7
    assert report['max_H']<1e-7
    assert report['max_power_error']<1e-7
    assert report['inverse_residual']<1e-11
    assert report['min_abs_j1_prime']>0


def test_modular_full_gradient_off_level(examples):
    module=examples['08'];rhs,H=module.modular_full_functions()
    with mp.workdps(55):
        values=np.array([complex(a) for a in module.modular_modes(.2)])*np.array([1.01,1,1,1])
        high=[mp.mpc(a) for a in values];a2=mp.mpf(float(np.log(3)))
        def original(*state):
            u1,u2,v1,v2=state;m=(1+u1*v1+u2*v2)/2
            bracket=mp.sinh(a2)*(u1*u1*u2*u2+v1*v1*v2*v2)-2*mp.cosh(a2)*u1*u2*v1*v2-(u1*v1-u2*v2-1)
            return 1j/mp.pi*mp.ellipk(m)**2*bracket
        grad=[mp.diff(lambda t:original(*(high[:j]+[t]+high[j+1:])),high[j]) for j in range(4)]
        expected=np.array([complex(grad[2]),complex(grad[3]),-complex(grad[0]),-complex(grad[1])])
        assert abs(H(values))>1e-7
        assert np.max(abs(rhs(0,values)-expected))<1e-10


def test_modular_theta_dh_and_continuation(examples):
    module=examples['08']
    assert all(v<1e-45 for v in module.modular_identity_checks().values())
    with mp.workdps(55):
        first=module.modular_modes(mp.mpf('.3'));later=module.modular_modes(mp.mpf('2.3'))
        assert max(abs(later[j]-(1j if j<2 else -1j)*first[j]) for j in range(4))<mp.mpf('1e-45')


def test_modular_trajectory(examples):
    report=examples['08'].modular_validation()['report']
    assert report['max_mode_error']<2e-7
    assert report['max_H']<2e-7
    assert report['max_invariant_error']<2e-7
    assert report['max_power_error']<2e-7


@pytest.mark.parametrize('number',['04','05','06','07','08'])
def test_example_notebook_execution_and_display_math(examples,number):
    outputs,_=examples[number].app.run()
    rendered=''.join(getattr(o,'text','') for o in outputs)
    assert 'class="arithmatex">||[' in rendered
    source=Path(examples[number].__file__).read_text()
    assert '$$' not in source
