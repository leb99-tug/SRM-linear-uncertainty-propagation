import re
import numpy as np


def extract_noise_floor_table(tex_file: str) -> np.ndarray:
    """
    Extract the numeric data table from a Noise_Floor.tex longtable into a numpy array.

    The .tex file uses European decimal notation (comma as separator, e.g. 10,000e+06).
    Data rows start after the \\endhead marker and end before \\end{longtable}.

    Returns
    -------
    np.ndarray, shape (n_frequencies, 37)
        Column 0  : frequency in Hz
        Columns 1-4   : Noise Floor NL Sx (S1..S4) / dB
        Columns 5-8   : Trace Noise NH Sx Mag / dB
        Columns 9-12  : Trace Noise NH Sx Phase / deg
        Columns 13-16 : Noise Floor NL Test Rcv bx / dBm
        Columns 17-20 : Trace Noise NH Test Rcv bx Mag / dB
        Columns 21-24 : Trace Noise NH Test Rcv bx Phase / deg
        Columns 25-28 : Noise Floor NL Ref Rcv ax / dBm
        Columns 29-32 : Trace Noise NH Ref Rcv ax Mag / dB
        Columns 33-36 : Trace Noise NH Ref Rcv ax Phase / deg
    """
    with open(tex_file, encoding="utf-8") as f:
        content = f.read()

    # Isolate the body after \endhead and before \end{longtable}
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
        cells = [c.strip() for c in line.split("&")]
        # Convert European decimal comma to dot, then parse
        values = [float(c.replace(",", ".")) for c in cells]
        rows.append(values)

    return np.array(rows)
