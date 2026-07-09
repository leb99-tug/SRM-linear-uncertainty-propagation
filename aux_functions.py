import re
import metas_unclib as munc
import numpy as np
import skrf as rf


def _parse_number(cell: str) -> float:
    """
    Parse a European-formatted number cell from these .tex table exports.

    Thousands are grouped with a LaTeX thin space + comma (``\\,``, e.g.
    ``26\\,000,000`` in the frequency column) and the decimal separator is
    a plain comma.
    """
    cell = cell.strip().replace("\\,", "")
    return float(cell.replace(",", "."))


def _parse_longtable(tex_file: str) -> np.ndarray:
    """
    Parse the numeric body of a ``longtable`` .tex export into a numpy array.

    Data rows start after the \\endhead marker and end before
    \\end{longtable}. Each row is ``&``-separated and ends with ``\\\\``.
    """
    with open(tex_file, encoding="utf-8") as f:
        content = f.read()

    body_match = re.search(r"\\endhead(.*?)\\end\{longtable\}", content, re.DOTALL)
    if body_match is None:
        raise ValueError("Could not find \\endhead ... \\end{longtable} in the file.")
    body = body_match.group(1)

    rows = []
    for line in body.splitlines():
        line = line.strip()
        # Data rows end with \\
        if not line.endswith("\\\\"):
            continue
        line = line[:-2]  # strip trailing \\
        cells = line.split("&")
        rows.append([_parse_number(c) for c in cells])

    return np.array(rows)


def extract_table(tex_file: str) -> np.ndarray:
    """
    Extract the numeric data table from a Noise_Floor.tex longtable into a numpy array.

    The .tex file uses European decimal notation (comma as separator, e.g. 10,000e+06).
    Data rows start after the \\endhead marker and end before \\end{longtable}.

    """
    return _parse_longtable(tex_file)


def interpolate_to_freq(
    values: np.ndarray,
    src_freq: np.ndarray,
    target_freq,
) -> np.ndarray:
    """
    Interpolate ``values`` from its own frequency grid onto ``target_freq``.

    Parameters
    ----------
    values : np.ndarray, shape (N,) or (N, M)
        Data to interpolate.  Column 0 must NOT be the frequency column —
        pass only the value columns (e.g. ``noise_data[:, 1]`` or
        ``noise_data[:, 1:5]``).
    src_freq : np.ndarray, shape (N,)
        Frequencies in Hz corresponding to the rows of *values*
        (typically ``noise_data[:, 0]``).
    target_freq : skrf.Frequency or np.ndarray
        Target frequency grid.  Accepts a ``skrf.Frequency`` object
        (``freq`` in the notebook) or a plain Hz array.

    Returns
    -------
    np.ndarray, shape (K,) or (K, M)
        Values interpolated onto *target_freq*, same dtype as *values*.
    """
    f_target = target_freq.f if hasattr(target_freq, "f") else np.asarray(target_freq)

    values = np.asarray(values)
    if values.ndim == 1:
        return np.interp(f_target, src_freq, values)

    return np.column_stack(
        [np.interp(f_target, src_freq, values[:, col]) for col in range(values.shape[1])]
    )


def extract_switch_terms(tex_file: str) -> tuple[rf.Network, rf.Network]:
    """
    Extract forward/reverse switch terms from a Thru_with_switch.tex longtable.

    The table holds S1,1 / S1,2 / S2,1 / S2,2 (Mag, Phase/deg) followed by the
    raw switch-term ratios ``a2/b2_p1`` (source at port 1, i.e. the forward
    switch term) and ``a1/b1_p2`` (source at port 2, i.e. the reverse switch
    term), each as a Mag/Phase(deg) pair. Frequency is in MHz.

    Returns
    -------
    forward, reverse : skrf.Network
        1-port-like networks whose ``s`` holds the complex switch-term ratio
        vs. frequency, e.g. for use as ``switch_term=(forward, reverse)`` in
        SRM/ALTOS.
    """
    table = _parse_longtable(tex_file)

    freq = rf.Frequency.from_f(table[:, 0], unit="MHz")

    def _to_complex(mag_col: int, phase_col: int) -> np.ndarray:
        return table[:, mag_col] * np.exp(1j * np.deg2rad(table[:, phase_col]))

    forward = _to_complex(9, 10)   # a2/b2_p1
    reverse = _to_complex(11, 12)  # a1/b1_p2

    forward_ntwk = rf.Network(frequency=freq, s=forward.reshape(-1, 1, 1), name="switch_term_forward")
    reverse_ntwk = rf.Network(frequency=freq, s=reverse.reshape(-1, 1, 1), name="switch_term_reverse")

    return forward_ntwk, reverse_ntwk

def s2t(S, pseudo=False):
    """Convert 2x2 S-parameters to T-parameters."""
    T = S.copy()
    T[0,0] = -(S[0,0]*S[1,1]-S[0,1]*S[1,0])
    T[0,1] = S[0,0]
    T[1,0] = -S[1,1]
    T[1,1] = 1
    return T if pseudo else T/S[1,0]

def t2s(T, pseudo=False):
    """Convert 2x2 T-parameters to S-parameters."""
    S = T.copy()
    S[0,0] = T[0,1]
    S[0,1] = T[0,0]*T[1,1]-T[0,1]*T[1,0]
    S[1,0] = 1
    S[1,1] = -T[1,0]
    return S if pseudo else S/T[1,1]

def embed_contributor(contrib_S, A_T, B_T, k):
    """
    Embed an uncertain contributor matrix (S-parameters) through the
    estimated error boxes A, B (T-parameters) and return the synthesized
    raw measurement (S-parameters, ucomplex 2x2).

        M_raw = t2s( k * A_T @ s2t(contrib_S) @ B_T )

    Parameters
    ----------
    contrib_S : (2,2) ucomplex
        Contributor at the calibrated plane, S-parameters.
    A_T, B_T : (2,2) complex
        Estimated error boxes (T-parameters, without uncertainty).
    k : complex
        Transmission tracking term.
    """
    return t2s(k * A_T @ s2t(contrib_S) @ B_T)


def cov_of_2x2(M_uc):
    """
    Real-valued covariance of a 2x2 ucomplex matrix.

    Element order: column-major vec -> [M00, M10, M01, M11], each element
    as (Re, Im) per the Metas convention. Returns an 8x8 real matrix.

    This is the contribution U_I of this contributor at the raw plane
    (eq. 21).
    """
    flat = M_uc.reshape(-1, order="F")          # column-major vec
    return np.asarray(munc.get_covariance(flat))


def inv_propagate_all_contributors(contribs_A, contribs_B, A_T, B_T, k,mid_S=None):
    """
    Two-port inverse propagation with contributors on BOTH ports and a
    central two-port (DUT / thru / reciprocal) in the middle:

        M_raw = t2s( k * A @ [prod contribs_A] @ s2t(mid) @ [prod contribs_B] @ B )

    Parameters
    ----------
    contribs_A : list of (2,2) ucomplex S-matrices for port A, ordered from
        the error box toward the middle (e.g. [D_A, C_A, R_A]).
    contribs_B : list of (2,2) ucomplex S-matrices for port B, ordered from
        the middle toward the error box (the physical mirror of A,
        typically [R_B, C_B, D_B]).
    mid_S : central two-port (S-parameters); None -> ideal thru (identity
        in the T-domain -> nominally M_raw = t2s(k A B)).

    """
    T = A_T.astype(object)                       # T-domain, holds ucomplex
    for cS in contribs_A:
        T = T @ s2t(cS)
    if mid_S is not None:
        T = T @ s2t(mid_S)
    for cS in contribs_B:
        T = T @ s2t(cS)
    M_raw = t2s(k * T @ B_T)
    return M_raw


def inv_propagate_one_port(contribs_S, rho_ideal, err_box_T, port="A"):
    """
    Inverse propagation for a ONE-PORT reflection standard (short/open/load).

    Parameters
    ----------
    contribs_S : list of (2,2) ucomplex S-matrices [D, C, R, ...] (nominally
        thru). Ordered from the error-box end to the standard end of the
        cascade.
    rho_ideal : ucomplex
        Ideal reflection (e.g. -1 short, +1 open, 0 load), optionally
        carrying sigma_S.
    err_box_T : (2,2)
        a[i] for port A or b[i] for port B (T-parameters).
    port : "A" (S11) or "B" (S22).

    Notes
    -----
    D is asymmetric (directivity/match). For port B, pass the port-B
    contributor matrices and/or the reversed contributor order as needed;
    C and R are symmetric and therefore order-independent.
    """
    G = s2t(contribs_S[0])
    for cS in contribs_S[1:]:                     # cascade contributors in the T-domain
        G = G @ s2t(cS)

    if port == "A":
        # raw1 -> A -> contributor -> standard; standard at port 2, looking into port 1
        E = t2s(err_box_T @ G)
        return E[0, 0] + E[0, 1] * E[1, 0] * rho_ideal / (1 - E[1, 1] * rho_ideal)
    elif port == "B":
        # standard -> contributor -> B -> raw2; standard at port 1, looking into port 2
        E = t2s(G @ err_box_T)
        return E[1, 1] + E[0, 1] * E[1, 0] * rho_ideal / (1 - E[0, 0] * rho_ideal)
    else:
        raise ValueError("port must be 'A' or 'B'")


def cov_of_scalar(z_uc):
    """
    Real-valued 2x2 covariance [[var_re, cov], [cov, var_im]] of a scalar
    ucomplex quantity -- exactly the format that
    ucomplex(..., covariance=...) expects (round-trips). For assigning to
    an S11/S22 measurement.
    """
    return np.asarray(munc.get_covariance(z_uc))


def forward_one_port(contribs_raw, gamma_meas, port="A"):
    """
    Vorwaerts-Propagation fuer eine EINTOR-Rohmessung (Reflexion).
    Belastet die gemessene Reflexion durch die Roh-Plane-Contributor:
 
        Port A:  Gamma'_raw = G11 + G12*G21*gamma / (1 - G22*gamma)
        Port B:  Gamma'_raw = G22 + G12*G21*gamma / (1 - G11*gamma)
        G = t2s( s2t(N) @ s2t(L) @ ... )
 
    contribs_raw : Liste (2,2) ucomplex S-Matrizen [N, L, ...] (nominal Thru).
    gamma_meas   : gemessene Reflexion -- komplex (nominal) oder ucomplex.
    port         : "A" (S11) oder "B" (S22).  Port B liest die andere Seite des
                   Contributor-Zweitors ab -- aequivalent zu einem gespiegelten
                   P @ N @ P, relevant bei asymmetrischem N (Noise-Floor/Trace).
 
    Rueckgabe: skalare ucomplex Gamma'_raw.  Kovarianz via cov_of_scalar -> U_F.
    Nominell (Contributor = Thru) ist Gamma'_raw = gamma_meas fuer beide Ports.
    """
    G = s2t(contribs_raw[0])
    for cS in contribs_raw[1:]:
        G = G @ s2t(cS)
    G = t2s(G)
    if port == "A":
        return G[0, 0] + G[0, 1] * G[1, 0] * gamma_meas / (1 - G[1, 1] * gamma_meas)
    elif port == "B":
        return G[1, 1] + G[0, 1] * G[1, 0] * gamma_meas / (1 - G[0, 0] * gamma_meas)
    else:
        raise ValueError("port must be 'A' or 'B'")

def forward_two_port(contribs_raw, M_meas_S):
    """
    Vorwaerts-Propagation fuer eine ZWEITOR-Rohmessung.
 
        M'_raw = t2s( s2t(N) @ s2t(L) @ ... @ s2t(M_meas) )
 
    contribs_raw : Liste (2,2) ucomplex S-Matrizen [N, L, ...] (nominal Thru),
                   Reihenfolge Instrument-Seite -> Messung.
    M_meas_S     : (2,2) gemessene S-Matrix -- komplex (nominal) ODER ucomplex,
                   falls die Messung selbst schon eine Kovarianz traegt.
 
    Rueckgabe: (2,2) ucomplex M'_raw.  Kovarianz via cov_of_2x2 -> U_F.
    Nominell (Contributor = Thru) ist M'_raw = M_meas; nur die Kovarianz waechst.
    """
    T = s2t(contribs_raw[0])
    for cS in contribs_raw[1:]:
        T = T @ s2t(cS)
    T = T @ s2t(M_meas_S)
    return t2s(T)

def create_attenuator(att_db=40.0, frequency=None, z0=50.0, name=None):
    """Ideal matched attenuator as an skrf Network.

    att_db : attenuation in dB (positive number; 40 -> -40 dB / |S21|=0.01)
    frequency : rf.Frequency or None (defaults to 0-40 GHz, 1001 pts)
    """
    if frequency is None:
        frequency = rf.Frequency(0, 40, 1001, 'ghz')
    a  = 10 ** (-abs(att_db) / 20)          # linear |S21| = |S12|
    nf = len(frequency)
    s  = np.zeros((nf, 2, 2), dtype=complex)
    s[:, 0, 1] = s[:, 1, 0] = a             # transmission, reciprocal
    return rf.Network(frequency=frequency, s=s, z0=z0,
                      name=name or f'{att_db:.0f} dB attenuator')