"""
@author srm: Ziad (zi.hatab@gmail.com)
@editor u_srm: Lukas (l-ebner@gmx.net)
Implementation of the symmetric-reciprocal-match (SRM) calibration method
with linear uncertainty propagation via metas_unclib .
"""

import numpy as np
import skrf as rf
import metas_unclib as munc


# constants
c0 = 299792458  # speed of light in vacuum (m/s)
p  = np.array([[0, 1], [1, 0]])          # permutation matrix (deterministic)
up = munc.ucomplexarray([[0, 1], [1, 0]], covariance=np.zeros((8, 8)))  # permutation (uncertain type)


# ---------------------------------------------------------------------------
# Matrix / S-parameter helpers
# ---------------------------------------------------------------------------

def correct_switch_term(S, GF, GR):
    """
    Correct switch terms of measured S-parameters at a single frequency point.

    Parameters
    ----------
    S  : 2x2 array – measured S-parameters
    GF : scalar    – forward switch term  (sourced by port 1)
    GR : scalar    – reverse switch term  (sourced by port 2)
    """
    S_new = S.copy()
    denom = 1 - S[0, 1] * S[1, 0] * GF * GR
    S_new[0, 0] = (S[0, 0] - S[0, 1] * S[1, 0] * GF) / denom
    S_new[0, 1] = (S[0, 1] - S[0, 0] * S[0, 1] * GR) / denom
    S_new[1, 0] = (S[1, 0] - S[1, 1] * S[1, 0] * GF) / denom
    S_new[1, 1] = (S[1, 1] - S[0, 1] * S[1, 0] * GR) / denom
    return S_new


def s2t(S, pseudo=False):
    """Convert 2x2 S-parameters to T-parameters."""
    T = S.copy()
    T[0, 0] = -(S[0, 0] * S[1, 1] - S[0, 1] * S[1, 0])
    T[0, 1] =  S[0, 0]
    T[1, 0] = -S[1, 1]
    T[1, 1] =  1
    return T if pseudo else T / S[1, 0]


def t2s(T, pseudo=False):
    """Convert 2x2 T-parameters to S-parameters."""
    S = T.copy()
    S[0, 0] = T[0, 1]
    S[0, 1] = T[0, 0] * T[1, 1] - T[0, 1] * T[1, 0]
    S[1, 0] = 1
    S[1, 1] = -T[1, 0]
    return S if pseudo else S / T[1, 1]


def sqrt_unwrapped(z):
    """Square root of a complex number with unwrapped phase."""
    return np.sqrt(abs(z)) * np.exp(0.5j * np.unwrap(np.angle(z)))


def input_reflection_l2r(T, G):
    return (T[0, 1] + T[0, 0] * G) / (T[1, 1] + T[1, 0] * G)


def inverse_reflection_l2r(T, G):
    return (G * T[1, 1] - T[0, 1]) / (T[0, 0] - T[1, 0] * G)


def input_reflection_r2l(T, G):
    return (T[0, 0] * G - T[1, 0]) / (T[1, 1] - T[0, 1] * G)


def inverse_reflection_r2l(T, G):
    return (G * T[1, 1] + T[1, 0]) / (T[0, 0] + T[0, 1] * G)


# ---------------------------------------------------------------------------
# Covariance helper
# ---------------------------------------------------------------------------

def convert2cov(x, num_f, cov_length=2):
    """
    Reshape/broadcast an uncertainty specification into a (num_f, cov_length, cov_length) array.

    Accepted input shapes
    ---------------------
    scalar   → diagonal matrix with that variance, tiled over frequencies
    1-D vec  → diagonalised, tiled over frequencies
    2-D mat  → tiled over frequencies
    3-D arr  → returned as-is (assumed fully specified)
    """
    num_f      = int(num_f)
    cov_length = int(cov_length)
    x = np.atleast_1d(x)
    if x.ndim > 2:
        return x                                          # fully specified
    if x.ndim == 2:
        return np.tile(x, (num_f, 1, 1))
    if x.shape[0] > 1:
        return np.tile(np.diag(x), (num_f, 1, 1))
    return np.tile(np.eye(cov_length) * x[0], (num_f, 1, 1))


# ---------------------------------------------------------------------------
# Core solvers – deterministic
# ---------------------------------------------------------------------------

def solve_symmetric_mobius(GammaA, GammaB):
    """
    Solve for the Möbius transformation H that maps port-B measurements to
    port-A measurements for symmetric one-port standards.

    See eq. (14) in the paper.

    Parameters
    ----------
    GammaA, GammaB : array-like – measured reflection coefficients at port A/B
    """
    G = np.array([[-b, -1, b * a, a] for a, b in zip(GammaA, GammaB)])
    _, _, vh = np.linalg.svd(G)
    return vh[-1, :].conj().reshape((2, 2), order='C')


def solve_box(W, Gamma, rho=0, port_A=True):
    """
    Solve for error-box coefficients from eigenvectors W and one match
    measurement (Gamma, rho).

    Parameters
    ----------
    W      : 2x2 eigenvector matrix
    Gamma  : measured match reflection coefficient
    rho    : defined (ideal) match reflection coefficient
    port_A : True → solve port-A box (A), False → solve port-B box (B)
    """
    w11 = W[0, 0] / W[1, 0]
    w12 = W[0, 1] / W[1, 1]
    if port_A:
        H = np.array([[-1,   -1,  w11,  w11],
                      [ 1,   -1, -w12,  w12],
                      [-rho, -1,  rho * Gamma, Gamma]])
    else:
        H = np.array([[-1,   -1,  w11,  w11],
                      [ 1,   -1, -w12,  w12],
                      [-rho,  1, -rho * Gamma, Gamma]])
    _, _, vh = np.linalg.svd(H)
    ns = vh[-1, :].conj()
    ns = ns / ns[-1]
    return ns.reshape((2, 2), order='C' if port_A else 'F')


def solve_at_one_freq(symmetric, est_symmetric, reciprocal, est_reciprocal,
                      reciprocal_GammaA, reciprocal_GammaB,
                      matchA, matchB, matchA_def, matchB_def, use_half_network):
    """
    Run one frequency point of the SRM calibration (deterministic).

    Parameters
    ----------
    symmetric        : (M, 2, 2) array  – measured S-params of M symmetric standards
    est_symmetric    : (M,) array       – estimated reflection coefficients (one per standard)
    reciprocal       : (2, 2) array     – measured S-params of reciprocal network
    est_reciprocal   : (2, 2) array     – estimated S-params of reciprocal network
    reciprocal_GammaA: list of M scalars – network-load Gamma from port A (NaN if not available)
    reciprocal_GammaB: list of M scalars – network-load Gamma from port B (NaN if not available)
    matchA, matchB   : scalars          – measured match reflection at port A / B
    matchA_def, matchB_def: scalars     – defined match reflection at port A / B
    use_half_network : bool             – use half-network formulation (on-wafer case)
    """
    H    = solve_symmetric_mobius(symmetric[:, 0, 0], symmetric[:, 1, 1])
    Hinv = np.linalg.inv(H)
    M_net = s2t(reciprocal)

    # Build virtual thru – use port A, port B, or average of both
    if np.isnan(reciprocal_GammaA[0]):
        Fb    = solve_symmetric_mobius(symmetric[:, 0, 0], reciprocal_GammaB)
        Fbinv = np.linalg.inv(Fb)
        M_thru = Fb @ Hinv @ M_net @ p @ Fbinv @ H @ p if use_half_network \
                 else M_net @ p @ Fbinv @ H @ p
    elif np.isnan(reciprocal_GammaB[0]):
        Fa    = solve_symmetric_mobius(reciprocal_GammaA, symmetric[:, 1, 1])
        Fainv = np.linalg.inv(Fa)
        M_thru = H @ Fainv @ M_net @ p @ Hinv @ Fa @ p if use_half_network \
                 else H @ Fainv @ M_net
    else:
        Fa    = solve_symmetric_mobius(reciprocal_GammaA, symmetric[:, 1, 1])
        Fainv = np.linalg.inv(Fa)
        M_thru_a = H @ Fainv @ M_net @ p @ Hinv @ Fa @ p if use_half_network \
                   else H @ Fainv @ M_net
        M_thru_a = M_thru_a / M_thru_a[-1, -1]

        Fb    = solve_symmetric_mobius(symmetric[:, 0, 0], reciprocal_GammaB)
        Fbinv = np.linalg.inv(Fb)
        M_thru_b = Fb @ Hinv @ M_net @ p @ Fbinv @ H @ p if use_half_network \
                   else M_net @ p @ Fbinv @ H @ p
        M_thru_b = M_thru_b / M_thru_b[-1, -1]

        M_thru = (M_thru_a + M_thru_b) / 2

    # Solve for error box A
    _, Wa = np.linalg.eig(M_thru @ p @ Hinv)
    A1   = solve_box(Wa,     matchA, rho=matchA_def, port_A=True)
    A2   = solve_box(Wa @ p, matchA, rho=matchA_def, port_A=True)
    err1 = abs(np.array([input_reflection_l2r(A1, e) - G
                         for G, e in zip(symmetric[:, 0, 0], est_symmetric)])).sum()
    err2 = abs(np.array([input_reflection_l2r(A2, e) - G
                         for G, e in zip(symmetric[:, 0, 0], est_symmetric)])).sum()
    A  = A1  if err1 < err2 else A2
    Wa = Wa  if err1 < err2 else Wa @ p

    # Solve for error box B
    _, Wb = np.linalg.eig((p @ Hinv @ M_thru).T)
    B1   = solve_box(Wb,     matchB, rho=matchB_def, port_A=False)
    B2   = solve_box(Wb @ p, matchB, rho=matchB_def, port_A=False)
    err1 = abs(np.array([input_reflection_r2l(B1, e) - G
                         for G, e in zip(symmetric[:, 1, 1], est_symmetric)])).sum()
    err2 = abs(np.array([input_reflection_r2l(B2, e) - G
                         for G, e in zip(symmetric[:, 1, 1], est_symmetric)])).sum()
    B  = B1  if err1 < err2 else B2
    Wb = Wb  if err1 < err2 else Wb @ p

    # Solve transmission term k
    k    = np.sqrt(np.linalg.det(s2t(reciprocal)) / np.linalg.det(A) / np.linalg.det(B))
    err1 = abs(k * A @ s2t(est_reciprocal) @ B - s2t(reciprocal)).sum()
    err2 = abs(k * A @ s2t(est_reciprocal) @ B + s2t(reciprocal)).sum()
    k    = k if err1 < err2 else -k

    return A, B, k, Wa, Wb


# ---------------------------------------------------------------------------
# Core solvers – uncertain (metas_unclib)
# ---------------------------------------------------------------------------

def u_solve_symmetric_mobius(GammaA, GammaB):
    # GammaA : list of input reflection from port-A
    # GammaB : list of input reflection from port-B
    # see equation (14) in the paper
    # GammaA : list of input reflection from port-A
    # GammaB : list of input reflection from port-B
    # see equation (14) in the paper
    #
    H = np.array([[-b, -1, b*a, a] for a, b in zip(GammaA, GammaB)])

   
    x3 = munc.ulinalg.lstsqrsolve(H[:, :-1], -H[:, -1])
    x = np.hstack([x3, 1.0])

    return x.reshape((2, 2), order='C')



def u_solve_box(W, Gamma, rho=0, port_A=True):
    """
    Uncertain version of solve_box using least-squares (lstsqrsolve) instead
    of SVD to keep the uncertainty propagation linear.

    Parameters – same as solve_box.
    """
    w11  = W[0, 0] / W[1, 0]
    w12  = W[0, 1] / W[1, 1]
    one  = munc.ucomplex(1,  covariance=np.zeros((2, 2), dtype=complex))
    mone = munc.ucomplex(-1, covariance=np.zeros((2, 2), dtype=complex))

    if port_A:
        H = np.array([[mone, mone,  w11,  w11],
                      [ one, mone, -w12,  w12],
                      [-rho, mone,  rho * Gamma, Gamma]])
    else:
        H = np.array([[mone, mone,  w11,  w11],
                      [ one, mone, -w12,  w12],
                      [-rho,  one, -rho * Gamma, Gamma]])

    # Fix the last unknown to 1, solve for the remaining three
    x_partial = munc.ulinalg.lstsqrsolve(H[:, :-1], -H[:, -1])
    ns = np.hstack([x_partial, 1.0])
    return ns.reshape((2, 2), order='C' if port_A else 'F')


def u_solve_at_one_freq(symmetric, est_symmetric, reciprocal, est_reciprocal,
                        reciprocal_GammaA, reciprocal_GammaB,
                        matchA, matchB, matchA_def, matchB_def,
                        use_half_network, GammaA_NAN=0, GammaB_NAN=0):
    """
    Uncertain version of solve_at_one_freq.

    Extra parameters
    ----------------
    GammaA_NAN : int – 1 if reciprocal_GammaA is unavailable (use port-B only)
    GammaB_NAN : int – 1 if reciprocal_GammaB is unavailable (use port-A only)
    """
    H    = u_solve_symmetric_mobius(symmetric[:, 0, 0], symmetric[:, 1, 1])
    Hinv = munc.ulinalg.inv(H)
    M_net = s2t(reciprocal)

    def _dot(*args):
        result = args[0]
        for m in args[1:]:
            result = munc.ulinalg.dot(result, m)
        return result

    # Build virtual thru
    if GammaA_NAN == 1:
        Fb    = u_solve_symmetric_mobius(symmetric[:, 0, 0], reciprocal_GammaB)
        Fbinv = munc.ulinalg.inv(Fb)
        M_thru = _dot(Fb, Hinv, M_net, up, Fbinv, H, up) if use_half_network \
                 else _dot(M_net, up, Fbinv, H, up)
    elif GammaB_NAN == 1:
        Fa    = u_solve_symmetric_mobius(reciprocal_GammaA, symmetric[:, 1, 1])
        Fainv = munc.ulinalg.inv(Fa)
        M_thru = _dot(H, Fainv, M_net, up, Hinv, Fa, up) if use_half_network \
                 else _dot(H, Fainv, M_net)
    else:
        Fa    = u_solve_symmetric_mobius(reciprocal_GammaA, symmetric[:, 1, 1])
        Fainv = munc.ulinalg.inv(Fa)
        M_thru_a = _dot(H, Fainv, M_net, up, Hinv, Fa, up) if use_half_network \
                   else _dot(H, Fainv, M_net)
        M_thru_a = M_thru_a / M_thru_a[-1, -1]

        Fb    = u_solve_symmetric_mobius(symmetric[:, 0, 0], reciprocal_GammaB)
        Fbinv = munc.ulinalg.inv(Fb)
        M_thru_b = _dot(Fb, Hinv, M_net, up, Fbinv, H, up) if use_half_network \
                   else _dot(M_net, up, Fbinv, H, up)
        M_thru_b = M_thru_b / M_thru_b[-1, -1]

        M_thru = (M_thru_a + M_thru_b) / 2

    # Solve for error box A
    Wa, _ = munc.ulinalg.eig(_dot(M_thru, up, Hinv))
    A1    = u_solve_box(Wa,                       matchA, rho=matchA_def, port_A=True)
    A2    = u_solve_box(munc.ulinalg.dot(Wa, up), matchA, rho=matchA_def, port_A=True)
    err1  = munc.umath.abs(np.array([input_reflection_l2r(A1, e) - G
                                     for G, e in zip(symmetric[:, 0, 0], est_symmetric)])).sum()
    err2  = munc.umath.abs(np.array([input_reflection_l2r(A2, e) - G
                                     for G, e in zip(symmetric[:, 0, 0], est_symmetric)])).sum()
    first = munc.get_value(err1) < munc.get_value(err2)
    A  = A1 if first else A2
    Wa = Wa if first else munc.ulinalg.dot(Wa, up)

    # Solve for error box B
    Wb, _ = munc.ulinalg.eig(_dot(up, Hinv, M_thru).T)
    B1    = u_solve_box(Wb,                       matchB, rho=matchB_def, port_A=False)
    B2    = u_solve_box(munc.ulinalg.dot(Wb, up), matchB, rho=matchB_def, port_A=False)
    err1  = munc.umath.abs(np.array([input_reflection_r2l(B1, e) - G
                                     for G, e in zip(symmetric[:, 1, 1], est_symmetric)])).sum()
    err2  = munc.umath.abs(np.array([input_reflection_r2l(B2, e) - G
                                     for G, e in zip(symmetric[:, 1, 1], est_symmetric)])).sum()
    first = munc.get_value(err1) < munc.get_value(err2)
    B  = B1 if first else B2
    Wb = Wb if first else munc.ulinalg.dot(Wb, up)

    # Solve transmission term k
    k    = munc.umath.sqrt(munc.ulinalg.det(s2t(reciprocal))
                           / munc.ulinalg.det(A) / munc.ulinalg.det(B))
    err1 = munc.umath.abs(_dot(A, s2t(est_reciprocal), B) * k - s2t(reciprocal)).sum()
    err2 = munc.umath.abs(_dot(A, s2t(est_reciprocal), B) * k + s2t(reciprocal)).sum()
    k    = k if munc.get_value(err1) < munc.get_value(err2) else -k

    return A, B, k, Wa, Wb


# ---------------------------------------------------------------------------
# uSRM class
# ---------------------------------------------------------------------------

class uSRM:
    """
    Symmetric-Reciprocal-Match (SRM) calibration with linear uncertainty
    propagation via metas_unclib.

    The interface mirrors srm.SRM as closely as possible.  Compared to SRM,
    each standard has a corresponding ``u<name>`` covariance argument.

    Parameters
    ----------
    symmetric11, symmetric12 : :class:`~skrf.network.Network` (1-port s11/s22)
        Port-A and port-B reflections of the 1st symmetric standard.
    symmetric21, symmetric22 : same for the 2nd symmetric standard.
    symmetric31, symmetric32 : same for the 3rd symmetric standard.
    est_symmetric1/2/3 : :class:`~skrf.network.Network` (2-port, diagonal)
        Prior estimates of the three symmetric standards.
    reciprocal : :class:`~skrf.network.Network` (2-port)
        Measured reciprocal/transmissive network.
    est_reciprocal : :class:`~skrf.network.Network` (2-port)
        Prior estimate of the reciprocal network.
    matchA, matchB : :class:`~skrf.network.Network` (1-port)
        Measured match standards at port A / B.
    matchA_def, matchB_def : :class:`~skrf.network.Network` (1-port), optional
        Defined (ideal) match values.  Defaults to 0.
    reciprocal_GammaA1/2/3 : :class:`~skrf.network.Network` (1-port), optional
        Network-load measurements at port A for each symmetric standard.
    reciprocal_GammaB1/2/3 : :class:`~skrf.network.Network` (1-port), optional
        Network-load measurements at port B for each symmetric standard.
    switch_term_fw, switch_term_bw : :class:`~skrf.network.Network` (1-port), optional
        Forward and reverse switch terms.
    use_half_network : bool, optional
        Use half-network formulation for on-wafer (fixed probe-distance) setups.

    Uncertainty arguments (all optional, default = 0 variance)
    -----------------------------------------------------------
    u<name> : scalar, 1-D, 2-D, or 3-D array
        Covariance of the corresponding measurement.  Interpreted by
        ``convert2cov`` – a scalar sets a uniform diagonal covariance.
    """

    def __init__(self,
                 # --- measured symmetric standards ---
                 symmetric,
                 # --- prior estimates ---
                 est_symmetric,
                 # --- reciprocal network ---
                 reciprocal, est_reciprocal,
                 # --- match standards ---
                 matchA, matchB,
                 matchA_def=None, matchB_def=None,
                 # --- network-load reflections ---
                 reciprocal_GammaA=None, reciprocal_GammaB=None,
                 # --- switch terms ---
                 switch_term_fw=None, switch_term_bw=None,
                 # --- options ---
                 use_half_network=False, model_fit=None,
                 # --- uncertainties ---
                 usymmetric=None,
                 ureciprocal=None,
                 umatchA=None,     umatchB=None,
                 umatchA_def=None, umatchB_def=None,
                 ureciprocal_GammaA=None,
                 ureciprocal_GammaB=None,
                 usw_fw=None, usw_bw=None):

        self.f = symmetric[0].frequency.f

        # Symmetric standard S-parameter data (port-A and port-B reflection each)
        self.Ssymmetric11 = symmetric[0].s.squeeze()
        self.Ssymmetric12 = symmetric[1].s.squeeze()
        self.Ssymmetric21 = symmetric[2].s.squeeze()
        self.Ssymmetric22 = symmetric[3].s.squeeze()
        self.Ssymmetric31 = symmetric[4].s.squeeze()
        self.Ssymmetric32 = symmetric[5].s.squeeze()

        # Prior estimates
        self.Sest_symmetric1 = est_symmetric[0].s.squeeze()
        self.Sest_symmetric2 = est_symmetric[1].s.squeeze()
        self.Sest_symmetric3 = est_symmetric[2].s.squeeze()

        # Reciprocal network
        self.Sreciprocal     = reciprocal.s.squeeze()
        self.Sest_reciprocal = est_reciprocal.s.squeeze()

        # Match standards
        self.SmatchA     = matchA.s.squeeze()
        self.SmatchB     = matchB.s.squeeze()
        self.SmatchA_def = self.SmatchA * 0 if matchA_def is None else matchA_def.s.squeeze()
        self.SmatchB_def = self.SmatchB * 0 if matchB_def is None else matchB_def.s.squeeze()

        # Network-load reflections (NaN-fill the missing port)
        self.GammaA_NAN = 0
        self.GammaB_NAN = 0
        if reciprocal_GammaA is None:
            self.Sreciprocal_GammaB1 = reciprocal_GammaB[0].s.squeeze()
            self.Sreciprocal_GammaB2 = reciprocal_GammaB[1].s.squeeze()
            self.Sreciprocal_GammaB3 = reciprocal_GammaB[2].s.squeeze()
            self.Sreciprocal_GammaA1 = self.Sreciprocal_GammaB1 * np.nan
            self.Sreciprocal_GammaA2 = self.Sreciprocal_GammaB2 * np.nan
            self.Sreciprocal_GammaA3 = self.Sreciprocal_GammaB3 * np.nan
            self.GammaA_NAN = 1
        elif reciprocal_GammaB is None:
            self.Sreciprocal_GammaA1 = reciprocal_GammaA[0].s.squeeze()
            self.Sreciprocal_GammaA2 = reciprocal_GammaA[1].s.squeeze()
            self.Sreciprocal_GammaA3 = reciprocal_GammaA[2].s.squeeze()
            self.Sreciprocal_GammaB1 = self.Sreciprocal_GammaA1 * np.nan
            self.Sreciprocal_GammaB2 = self.Sreciprocal_GammaA2 * np.nan
            self.Sreciprocal_GammaB3 = self.Sreciprocal_GammaA3 * np.nan
            self.GammaB_NAN = 1
        else:
            self.Sreciprocal_GammaA1 = reciprocal_GammaA[0].s.squeeze()
            self.Sreciprocal_GammaA2 = reciprocal_GammaA[1].s.squeeze()
            self.Sreciprocal_GammaA3 = reciprocal_GammaA[2].s.squeeze()
            self.Sreciprocal_GammaB1 = reciprocal_GammaB[0].s.squeeze()
            self.Sreciprocal_GammaB2 = reciprocal_GammaB[1].s.squeeze()
            self.Sreciprocal_GammaB3 = reciprocal_GammaB[2].s.squeeze()

        # Switch terms (default to zero)
        if switch_term_fw is not None and switch_term_bw is not None:
            self.switch_term_fw = switch_term_fw.s.squeeze()
            self.switch_term_bw = switch_term_bw.s.squeeze()
        else:
            self.switch_term_fw = self.f * 0
            self.switch_term_bw = self.f * 0

        self.use_half_network = use_half_network
        self.model_fit        = model_fit

        # Uncertainty covariances (default to zero variance)
        self.usymmetric11 = usymmetric[0] if usymmetric[0] is not None else 0
        self.usymmetric12 = usymmetric[1] if usymmetric[1] is not None else 0
        self.usymmetric21 = usymmetric[2] if usymmetric[2] is not None else 0
        self.usymmetric22 = usymmetric[3] if usymmetric[3] is not None else 0
        self.usymmetric31 = usymmetric[4] if usymmetric[4] is not None else 0
        self.usymmetric32 = usymmetric[5] if usymmetric[5] is not None else 0
        self.ureciprocal  = ureciprocal  if ureciprocal  is not None else 0
        self.umatchA      = umatchA      if umatchA      is not None else 0
        self.umatchB      = umatchB      if umatchB      is not None else 0
        self.umatchA_def  = umatchA_def  if umatchA_def  is not None else 0
        self.umatchB_def  = umatchB_def  if umatchB_def  is not None else 0
        self.ureciprocal_GammaA1 = ureciprocal_GammaA[0] if ureciprocal_GammaA is not None else 0
        self.ureciprocal_GammaA2 = ureciprocal_GammaA[1] if ureciprocal_GammaA is not None else 0
        self.ureciprocal_GammaA3 = ureciprocal_GammaA[2] if ureciprocal_GammaA is not None else 0
        self.ureciprocal_GammaB1 = ureciprocal_GammaB[0] if ureciprocal_GammaB is not None else 0
        self.ureciprocal_GammaB2 = ureciprocal_GammaB[1] if ureciprocal_GammaB is not None else 0
        self.ureciprocal_GammaB3 = ureciprocal_GammaB[2] if ureciprocal_GammaB is not None else 0
        self.uswitch_term_fw = usw_fw if usw_fw is not None else 0
        self.uswitch_term_bw = usw_bw if usw_bw is not None else 0

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _build_symmetric_array(self, s11, s12, s21, s22, s31, s32):
        """Stack six scalar reflections into a (3, 2, 2) array."""
        sym = np.zeros((3, 2, 2), dtype=type(s11))
        sym[0, 0, 0] = s11;  sym[0, 1, 1] = s12
        sym[1, 0, 0] = s21;  sym[1, 1, 1] = s22
        sym[2, 0, 0] = s31;  sym[2, 1, 1] = s32
        return sym

    def _build_est_symmetric_array(self, e1, e2, e3, dtype=complex):
        """Stack three 2x2 estimate arrays into a (3, 2, 2) array."""
        est = np.zeros((3, 2, 2), dtype=dtype)
        est[0] = e1;  est[1] = e2;  est[2] = e3
        return est

    # -----------------------------------------------------------------------
    # Public: deterministic run
    # -----------------------------------------------------------------------

    def run(self):
        """Run the SRM calibration (deterministic, no uncertainty)."""
        A, B, k, Wa, Wb = [], [], [], [], []

        print('\nSRM is running...')
        for inx, f in enumerate(self.f):
            symmetric = self._build_symmetric_array(
                self.Ssymmetric11[inx], self.Ssymmetric12[inx],
                self.Ssymmetric21[inx], self.Ssymmetric22[inx],
                self.Ssymmetric31[inx], self.Ssymmetric32[inx])

            est_symmetric = [self.Sest_symmetric1[inx],
                             self.Sest_symmetric2[inx],
                             self.Sest_symmetric3[inx]]

            reciprocal_GammaA = [self.Sreciprocal_GammaA1[inx],
                                 self.Sreciprocal_GammaA2[inx],
                                 self.Sreciprocal_GammaA3[inx]]
            reciprocal_GammaB = [self.Sreciprocal_GammaB1[inx],
                                 self.Sreciprocal_GammaB2[inx],
                                 self.Sreciprocal_GammaB3[inx]]

            reciprocal     = correct_switch_term(self.Sreciprocal[inx],
                                                 self.switch_term_fw[inx],
                                                 self.switch_term_bw[inx])

            A_, B_, k_, Wa_, Wb_ = solve_at_one_freq(
                symmetric=symmetric,
                est_symmetric=est_symmetric,
                reciprocal=reciprocal,
                est_reciprocal=self.Sest_reciprocal[inx],
                reciprocal_GammaA=reciprocal_GammaA,
                reciprocal_GammaB=reciprocal_GammaB,
                matchA=self.SmatchA[inx],    matchB=self.SmatchB[inx],
                matchA_def=self.SmatchA_def[inx], matchB_def=self.SmatchB_def[inx],
                use_half_network=self.use_half_network)

            A.append(A_);  B.append(B_);  k.append(k_)
            Wa.append(Wa_);  Wb.append(Wb_)
            #print(f'Frequency: {f * 1e-9:.2f} GHz ... DONE!')

        self.A  = np.array(A)
        self.B  = np.array(B)
        self.X  = np.array([np.kron(b.T, a) for a, b in zip(self.A, self.B)])
        self.k  = np.array(k)
        self.Wa = np.array(Wa)
        self.Wb = np.array(Wb)
        self.error_coef()

    # -----------------------------------------------------------------------
    # Public: uncertain run
    # -----------------------------------------------------------------------

    def u_run(self):
        """Run the SRM calibration with linear uncertainty propagation."""
        A, B, k, Wa, Wb = [], [], [], [], []

        # Expand all covariance specs to (num_f, n, n) arrays
        nf = len(self.f)
        usym11 = convert2cov(self.usymmetric11, nf, 2)
        usym12 = convert2cov(self.usymmetric12, nf, 2)
        usym21 = convert2cov(self.usymmetric21, nf, 2)
        usym22 = convert2cov(self.usymmetric22, nf, 2)
        usym31 = convert2cov(self.usymmetric31, nf, 2)
        usym32 = convert2cov(self.usymmetric32, nf, 2)
        urecip = convert2cov(self.ureciprocal,  nf, 8)
        usw_fw = convert2cov(self.uswitch_term_fw, nf, 2)
        usw_bw = convert2cov(self.uswitch_term_bw, nf, 2)
        ugA1   = convert2cov(self.ureciprocal_GammaA1, nf, 2)
        ugA2   = convert2cov(self.ureciprocal_GammaA2, nf, 2)
        ugA3   = convert2cov(self.ureciprocal_GammaA3, nf, 2)
        ugB1   = convert2cov(self.ureciprocal_GammaB1, nf, 2)
        ugB2   = convert2cov(self.ureciprocal_GammaB2, nf, 2)
        ugB3   = convert2cov(self.ureciprocal_GammaB3, nf, 2)
        umA    = convert2cov(self.umatchA,     nf, 2)
        umB    = convert2cov(self.umatchB,     nf, 2)
        umA_d  = convert2cov(self.umatchA_def, nf, 2)
        umB_d  = convert2cov(self.umatchB_def, nf, 2)
        zero2  = np.zeros((2, 2), dtype=complex)

        print('\nuSRM is running...')
        for inx, f in enumerate(self.f):
            # Wrap scalars as uncertain complexes with given covariance
            def uc(val, cov): return munc.ucomplex(val, covariance=cov)
            def uca(val, cov): return munc.ucomplexarray(values=val, covariance=cov)
            zero8 = np.zeros((8, 8), dtype=complex)

            s11 = uc(self.Ssymmetric11[inx], usym11[inx])
            s12 = uc(self.Ssymmetric12[inx], usym12[inx])
            s21 = uc(self.Ssymmetric21[inx], usym21[inx])
            s22 = uc(self.Ssymmetric22[inx], usym22[inx])
            s31 = uc(self.Ssymmetric31[inx], usym31[inx])
            s32 = uc(self.Ssymmetric32[inx], usym32[inx])

            symmetric = self._build_symmetric_array(s11, s12, s21, s22, s31, s32)

            est_symmetric = self._build_est_symmetric_array(
                self.Sest_symmetric1[inx],
                self.Sest_symmetric2[inx],
                self.Sest_symmetric3[inx],
                dtype=munc.ucomplex)
            # Convert estimates to uncertain scalars (zero variance – they are priors)
            est_s = [uc(e[0, 0], zero2) for e in
                     [self.Sest_symmetric1[inx],
                      self.Sest_symmetric2[inx],
                      self.Sest_symmetric3[inx]]]

            reciprocal     = uca(self.Sreciprocal[inx],     urecip[inx])
            est_reciprocal = uca(self.Sest_reciprocal[inx], zero8)

            reciprocal_GammaA = [uc(self.Sreciprocal_GammaA1[inx], ugA1[inx]),
                                 uc(self.Sreciprocal_GammaA2[inx], ugA2[inx]),
                                 uc(self.Sreciprocal_GammaA3[inx], ugA3[inx])]
            reciprocal_GammaB = [uc(self.Sreciprocal_GammaB1[inx], ugB1[inx]),
                                 uc(self.Sreciprocal_GammaB2[inx], ugB2[inx]),
                                 uc(self.Sreciprocal_GammaB3[inx], ugB3[inx])]

            matchA     = uc(self.SmatchA[inx],     umA[inx])
            matchB     = uc(self.SmatchB[inx],     umB[inx])
            matchA_def = uc(self.SmatchA_def[inx], umA_d[inx])
            matchB_def = uc(self.SmatchB_def[inx], umB_d[inx])

            sw_fw = uc(self.switch_term_fw[inx], usw_fw[inx])
            sw_bw = uc(self.switch_term_bw[inx], usw_bw[inx])
            reciprocal = correct_switch_term(reciprocal, sw_fw, sw_bw)

            A_, B_, k_, Wa_, Wb_ = u_solve_at_one_freq(
                symmetric=symmetric,
                est_symmetric=est_s,
                reciprocal=reciprocal,
                est_reciprocal=est_reciprocal,
                reciprocal_GammaA=reciprocal_GammaA,
                reciprocal_GammaB=reciprocal_GammaB,
                matchA=matchA, matchB=matchB,
                matchA_def=matchA_def, matchB_def=matchB_def,
                use_half_network=self.use_half_network,
                GammaA_NAN=self.GammaA_NAN,
                GammaB_NAN=self.GammaB_NAN)

            A.append(A_);  B.append(B_);  k.append(k_)
            Wa.append(Wa_);  Wb.append(Wb_)
            #print(f'Frequency: {f * 1e-9:.2f} GHz ... DONE!')

        self.A  = np.array(A)
        self.B  = np.array(B)
        self.X  = np.array([np.kron(b.T, a) for a, b in zip(self.A, self.B)])
        self.k  = np.array(k)
        self.Wa = np.array(Wa)
        self.Wb = np.array(Wb)

    # -----------------------------------------------------------------------
    # Apply calibration
    # -----------------------------------------------------------------------

    def apply_cal(self, NW, left=True):
        """
        Apply calibration (deterministic) to a 1- or 2-port network.

        Parameters
        ----------
        NW   : :class:`~skrf.network.Network`
        left : bool – which port to return for 1-port networks
        """
        nports = NW.s.shape[-1] if hasattr(NW, 's') else 2
        if nports < 2:
            NW = rf.two_port_reflect(NW)

        S_cal = []
        for x, k, s, swf, swb in zip(self.X, self.k, NW.s,
                                      self.switch_term_fw, self.switch_term_bw):
            s    = correct_switch_term(s, swf, swb) if np.any([swf, swb]) else s
            xinv = np.linalg.pinv(x)
            M_   = np.array([-s[0, 0] * s[1, 1] + s[0, 1] * s[1, 0], -s[1, 1], s[0, 0], 1])
            T_   = xinv @ M_
            s21  = k * s[1, 0] / T_[-1] + np.finfo(float).eps
            T_   = T_ / T_[-1]
            S_cal.append([[T_[2], (T_[0] - T_[2] * T_[1]) / s21], [s21, -T_[1]]])

        S_cal = np.array(S_cal).squeeze()
        if nports < 2:
            S_cal = S_cal[:, 0, 0] if left else S_cal[:, 1, 1]
        return rf.Network(frequency=NW.frequency, s=S_cal.squeeze())

    def u_apply_cal(self, NW, left=True):
        """
        Apply calibration (uncertain) to a 1- or 2-port network.

        Returns
        -------
        NW_cal : :class:`~skrf.network.Network` – calibrated network (mean values)
        unc    : ndarray – standard uncertainty of S-parameters
        """
        nports = NW.s.shape[-1] if hasattr(NW, 's') else 2
        if nports < 2:
            NW = rf.two_port_reflect(NW)

        S_cal = []
        for x, k, s, swf, swb in zip(self.X, self.k, NW.s,
                                      self.switch_term_fw, self.switch_term_bw):
            s    = correct_switch_term(s, swf, swb) if np.any([swf, swb]) else s
            xinv = munc.ulinalg.inv(x)
            M_   = np.array([-s[0, 0] * s[1, 1] + s[0, 1] * s[1, 0], -s[1, 1], s[0, 0], 1])
            T_   = munc.ulinalg.dot(xinv, M_)
            s21  = k * s[1, 0] / T_[-1] + np.finfo(float).eps
            T_   = T_ / T_[-1]
            S_cal.append([[T_[2], (T_[0] - T_[2] * T_[1]) / s21], [s21, -T_[1]]])

        S_cal = np.array(S_cal).squeeze()
        if nports < 2:
            S_cal = S_cal[:, 0, 0] if left else S_cal[:, 1, 1]
        S_cal = S_cal.squeeze()
        return (rf.Network(frequency=NW.frequency, s=munc.get_value(S_cal)),
                munc.get_stdunc(S_cal))

    # -----------------------------------------------------------------------
    # Error coefficients and derived quantities
    # -----------------------------------------------------------------------

    def error_coef(self):
        """
        Compute 12-term VNA error coefficients from the 8-term error-box model.

        References
        ----------
        [4] R. B. Marks, "Formulations of the Basic Vector Network Analyzer
            Error Model including Switch-Terms," 50th ARFTG Conf., 1997.
        [5] Dunsmore, J. P., Handbook of Microwave Component Measurements,
            Wiley, 2020.

        Forward (port-1 sourced): EDF, ESF, ERF, ELF, ETF, EXF
        Reverse (port-2 sourced): EDR, ESR, ERR, ELR, ETR, EXR
        Switch terms            : GF, GR
        """
        self.coefs = {}
        EDF =  self.X[:, 2, 3]
        ESF = -self.X[:, 3, 2]
        ERF =  self.X[:, 2, 2] - self.X[:, 2, 3] * self.X[:, 3, 2]

        EDR = -self.X[:, 1, 3]
        ESR =  self.X[:, 3, 1]
        ERR =  self.X[:, 1, 1] - self.X[:, 3, 1] * self.X[:, 1, 3]

        GF = self.switch_term_fw
        GR = self.switch_term_bw

        ELF = ESR + ERR * GF / (1 - EDR * GF)
        ETF = 1 / self.k / (1 - EDR * GF)
        EXF = 0 * ESR

        ELR = ESF + ERF * GR / (1 - EDF * GR)
        ETR = self.k * ERR * ERF / (1 - EDF * GR)
        EXR = 0 * ESR

        self.coefs.update({
            'EDF': EDF, 'ESF': ESF, 'ERF': ERF,
            'ELF': ELF, 'ETF': ETF, 'EXF': EXF, 'GF': GF,
            'EDR': EDR, 'ESR': ESR, 'ERR': ERR,
            'ELR': ELR, 'ETR': ETR, 'EXR': EXR, 'GR': GR,
            # Consistency check (eq. 35 in [4]) – should be ~0
            'check': abs(ETF * ETR
                         - (ERR + EDR * (ELF - ESR))
                         * (ERF + EDF * (ELR - ESF))),
        })
        return self.coefs

    def reciprocal_ntwk(self):
        """
        Return left and right error-boxes as skrf Networks (reciprocal assumption).
        """
        freq = rf.Frequency.from_f(self.f, unit='hz')
        freq.unit = 'ghz'

        def _make_ntwk(s11, s22, erf, name):
            s21 = sqrt_unwrapped(erf)
            S   = np.array([[[a, c], [c, b]]
                             for a, b, c in zip(s11, s22, s21)])
            return rf.Network(s=S, frequency=freq, name=name)

        left_ntwk  = _make_ntwk(self.coefs['EDF'], self.coefs['ESF'],
                                 self.coefs['ERF'], 'Left error-box')
        right_ntwk = _make_ntwk(self.coefs['EDR'], self.coefs['ESR'],
                                 self.coefs['ERR'], 'Right error-box')
        right_ntwk.flip()
        return left_ntwk, right_ntwk

    def renorm_impedance(self, Z_new, Z0=50):
        """
        Re-normalise the calibration reference impedance.

        Parameters
        ----------
        Z_new : scalar or array – new reference impedance
        Z0    : scalar or array – old reference impedance (default 50 Ω)
        """
        N     = len(self.k)
        Z_new = Z_new * np.ones(N)
        Z0    = Z0    * np.ones(N)
        G     = (Z_new - Z0) / (Z_new + Z0)

        X_new, K_new = [], []
        for x, k, g in zip(self.X, self.k, G):
            KX = k * x @ np.kron([[1, -g], [-g, 1]], [[1, g], [g, 1]]) / (1 - g ** 2)
            X_new.append(KX / KX[-1, -1])
            K_new.append(KX[-1, -1])

        self.X = np.array(X_new)
        self.k = np.array(K_new)


# EOF
