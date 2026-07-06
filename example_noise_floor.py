import matplotlib.pyplot as plt
import numpy as np

from extract_latex_table import extract_table

data = extract_table("Uncertainty_Contributor_Data/Noise_Floor.tex")

freq_GHz = data[:, 0] / 1e9

# --- column slices ----------------------------------------------------------
# Noise Floor NL
nf_s   = data[:, 1:5]      # S1-S4          / dB
nf_b   = data[:, 13:17]    # Test Rcv b1-b4 / dBm
nf_a   = data[:, 25:29]    # Ref Rcv a1-a4  / dBm

# Trace Noise NH – magnitude
tn_s_mag = data[:, 5:9]    # S1-S4          / dB
tn_b_mag = data[:, 17:21]  # Test Rcv b1-b4 / dB
tn_a_mag = data[:, 29:33]  # Ref Rcv a1-a4  / dB

# Trace Noise NH – phase
tn_s_ph  = data[:, 9:13]   # S1-S4          / deg
tn_b_ph  = data[:, 21:25]  # Test Rcv b1-b4 / deg
tn_a_ph  = data[:, 33:37]  # Ref Rcv a1-a4  / deg

# --- labels -----------------------------------------------------------------
s_labels  = ["S1", "S2", "S3", "S4"]
b_labels  = ["b1", "b2", "b3", "b4"]
a_labels  = ["a1", "a2", "a3", "a4"]

# --- plot -------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(15, 10), sharex=True)
fig.suptitle("Noise Floor & Trace Noise — Noise_Floor.tex", fontsize=13)

groups = [
    # (axes column, NF data, TN-mag data, TN-phase data, labels, NF y-label)
    (0, nf_s, tn_s_mag, tn_s_ph, s_labels, "Noise Floor NL / dB"),
    (1, nf_b, tn_b_mag, tn_b_ph, b_labels, "Noise Floor NL / dBm"),
    (2, nf_a, tn_a_mag, tn_a_ph, a_labels, "Noise Floor NL / dBm"),
]
col_titles = ["S-parameters (S1–S4)", "Test Receiver (b1–b4)", "Ref. Receiver (a1–a4)"]

for col, (ax_col, nf, tn_mag, tn_ph, labels, nf_ylabel) in enumerate(groups):
    for i, label in enumerate(labels):
        axes[0, ax_col].plot(freq_GHz, nf[:, i],     label=label)
        axes[1, ax_col].plot(freq_GHz, tn_mag[:, i], label=label)
        axes[2, ax_col].plot(freq_GHz, tn_ph[:, i],  label=label)

    axes[0, ax_col].set_title(col_titles[col])
    axes[0, ax_col].set_ylabel(nf_ylabel)
    axes[1, ax_col].set_ylabel("Trace Noise NH Mag / dB (k=1)")
    axes[2, ax_col].set_ylabel("Trace Noise NH Phase / ° (k=1)")
    axes[2, ax_col].set_xlabel("Frequency / GHz")

    for row in range(3):
        axes[row, ax_col].legend(fontsize=7)
        axes[row, ax_col].grid(True, linewidth=0.4)

plt.tight_layout()
plt.savefig("noise_floor_overview.png", dpi=150)
plt.show()
