import numpy as np

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
    return t[idx0:], v[:, idx0:]

ls = {
    'dt':    0.01,
    't1':    0.0,
    'trans': 5.0,
    't2':    50.0,
    'beta':  8/3,
    'sigma': 10.0,
    'eps':   [0.01, 0.01, 0.01],
}
# # constant-ρ series at 28.0
# rho_series = np.full(int((ls['t2']-ls['t1'])/ls['dt'])+1, 28.0)
# v0 = [1.0, 1.0, 1.0]

# t, v = get_lorenz(ls, rho_series, v0)
# print(v)


# # then v[0], v[1], v[2], v[3] are x, y, z, rho respectively