import re

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
