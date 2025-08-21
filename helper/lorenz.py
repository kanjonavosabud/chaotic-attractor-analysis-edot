import numpy as np
from reservoirpy.datasets import lorenz
from scipy.integrate import solve_ivp

def generate_lorenz_data(n_timesteps, sigma=10, rho=28, beta=8/3, h=0.02, x0=None):
    """Generate data from Lorenz system"""
    if x0 is None:
        x0 = [1, 1, 1]
    return lorenz(n_timesteps=n_timesteps, sigma=sigma, rho=rho, beta=beta, h=h, x0=x0)

def lorenz_system(t, state, sigma, beta, rho):
    """Lorenz system differential equations"""
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = (x * (rho - z)) - y
    dzdt = (x * y) - (beta * z)
    return [dxdt, dydt, dzdt]

def solve_lorenz(t_end=50, t_eval=0.1, initial_state=[1, 1, 1], sigma=10, beta=8/3, rho=28):
    """Solve Lorenz system using scipy's solve_ivp"""

    t_span = (0, t_end)  # Time span for evolution
    t_eval = np.arange(0, t_end, t_eval)  # Time points

    sol = solve_ivp(lorenz_system, t_span, initial_state, 
                    args=(sigma, beta, rho), t_eval=t_eval)
    data = sol.y.T
    return data


def get_lorenz(ls, rho, v0):
    """
    Generate Lorenz trajectory using RK4 method with additive noise.

    Parameters
    ----------
    ls : dict
        Must contain keys:
          - 'dt'    : float, time step
          - 't1'    : float, start time
          - 'trans' : float, transient length to discard
          - 't2'    : float, end time
          - 'beta'  : float, Lorenz β parameter
          - 'sigma' : float, Lorenz σ parameter
          - 'eps'   : array-like of length 3, max noise per channel
    rho : array_like, shape (n_steps,)
        Time series of ρ values (must match length of time vector).
    v0 : array_like, shape (3,)
        Initial state [x0, y0, z0].

    Returns
    -------
    t : ndarray, shape (n_steps_post,)
        Time points after discarding the transient.
    v : ndarray, shape (4, n_steps_post)
        Rows are [x; y; z; rho].
    """
    dt    = ls['dt']
    t1    = ls['t1']
    trans = ls['trans']
    t2    = ls['t2']
    beta  = ls['beta']
    sigma = ls['sigma']
    eps   = np.asarray(ls['eps'])

    # build time vector (inclusive of t2 if it lands exactly)
    t = np.arange(t1, t2 + dt, dt)
    n = t.size

    rho = np.asarray(rho)
    if rho.shape[0] != n:
        raise ValueError("Length of rho must equal number of time steps")

    # prepare storage
    v = np.zeros((3, n))
    v[:, 0] = v0

    # pre‑generate additive noise for each step
    noise = -eps[:, None] + 2.0 * eps[:, None] * np.random.rand(3, n)

    # Lorenz vector field
    def f(vi, rho_i):
        x, y, z = vi
        return np.array([
            sigma * (y - x),
            x * (rho_i - z) - y,
            x * y - beta * z
        ])

    # RK4 integration
    for i in range(n - 1):
        vi    = v[:, i]
        rho_i = rho[i]

        k1 = dt * f(vi,        rho_i)
        k2 = dt * f(vi + 0.5*k1, rho_i)
        k3 = dt * f(vi + 0.5*k2, rho_i)
        k4 = dt * f(vi +    k3, rho_i)

        v[:, i+1] = vi + (k1 + 2*k2 + 2*k3 + k4) / 6.0 \
                    + noise[:, i]

    # append rho as the 4th row
    v = np.vstack([v, rho])

    # discard transient
    idx0 = int(round(trans / dt))

    # return times and positions
    return t[idx0:], v[:, idx0:]

def my_lorenz(sigma=10, beta=8/3, rho=28, t_start=0, t_end=50, t_trans=0, t_eval=.01, x0=[1,1,1]):

    ls = {
        'dt':    t_eval,
        't1':    t_start,
        'trans': t_trans,
        't2':    t_end,
        'beta':  beta,
        'sigma': sigma,
        'eps':   [0, 0, 0],
    }

    # constant-ρ series at 28.0
    rho_series = np.full(int((ls['t2']-ls['t1'])/ls['dt'])+1, rho)
    v0 = x0

    t, v = get_lorenz(ls, rho_series, v0)
    data = v[:-1].T
    return data