"""
main_sim.py — Phase 3: BER Simulation & Waterfall Plot
=======================================================
Monte-Carlo BER simulation for a QPSK baseband transceiver over an AWGN
channel.  Sweeps Eb/N0 from 0 dB to 10 dB and compares empirical BER
against the theoretical closed-form expression.

Theoretical BER for QPSK over AWGN
------------------------------------
QPSK can be decomposed into two independent BPSK streams on the I and Q
arms.  Each arm's BER is identical to BPSK BER:

    Pb = Q(sqrt(2·Eb/N0))
       = 0.5 · erfc(sqrt(Eb/N0))

where:
    · Q(x) = 0.5 · erfc(x/√2)  is the Q-function (tail probability of N(0,1))
    · erfc(x) = (2/√π) · ∫_x^∞ exp(−t²) dt  is the complementary error function
    · Eb/N0 is the linear (not dB) energy-per-bit to noise-PSD ratio

References
----------
- Proakis & Salehi, "Digital Communications", 5th ed., §8.2.5, Eq. (8.2-34).
- Sklar, "Digital Communications", 2nd ed., §4.4.
- ETSI EN 302 307-1 V1.4.1 (2014-11), Annex B.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erfc

from channel import awgn_channel
from receiver import receive
from transmitter import transmit

# Use non-interactive backend when no display is available (e.g., CI servers)
if os.environ.get("DISPLAY") is None and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
N_BITS: int = 20_000          # information bits per Monte-Carlo trial
SPS: int = 8                  # samples per symbol
BETA: float = 0.35            # RRC roll-off factor (ETSI EN 302 307-1)
N_TAPS: int = 101             # RRC filter length (odd, ≈12.5 symbol periods)
BITS_PER_SYMBOL: int = 2      # QPSK: log2(4) = 2
EBN0_DB_RANGE = np.arange(0, 11, 1)   # 0 dB … 10 dB inclusive, step 1 dB
SEED: int = 42                # transmitter PRNG seed (fixed for repeatability)


# ---------------------------------------------------------------------------
# Helper: theoretical QPSK BER
# ---------------------------------------------------------------------------
def theoretical_ber(ebn0_db_array: np.ndarray) -> np.ndarray:
    """
    Compute the closed-form theoretical BER for QPSK over AWGN.

    Parameters
    ----------
    ebn0_db_array : ndarray of float
        Eb/N0 values in dB.

    Returns
    -------
    ber_theory : ndarray of float
        Theoretical BER for each Eb/N0 value.

    Mathematical model
    ------------------
    QPSK decomposes into two orthogonal BPSK channels (I and Q).  Each
    independent channel has symbol energy Es/2 = Eb (since k=2 bits/symbol).
    The per-channel BER equals the BPSK BER:

        Pb = Q(√(2·Eb/N0))

    Using the identity  Q(x) = 0.5 · erfc(x/√2):

        Pb = 0.5 · erfc(√(Eb/N0))

    This is the exact expression for Gray-coded QPSK under AWGN (no
    approximation); adjacent symbols always differ by exactly one bit, so
    symbol errors contribute ≈1 bit error each at moderate-to-high SNR.

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., §8.2.5, Eq. (8.2-34).
    """
    ebn0_linear = 10.0 ** (ebn0_db_array / 10.0)
    return 0.5 * erfc(np.sqrt(ebn0_linear))


# ---------------------------------------------------------------------------
# Monte-Carlo BER sweep
# ---------------------------------------------------------------------------
def run_ber_simulation(
    ebn0_db_range: np.ndarray,
    n_bits: int = N_BITS,
    sps: int = SPS,
    beta: float = BETA,
    n_taps: int = N_TAPS,
    seed: int = SEED,
) -> np.ndarray:
    """
    Run the Monte-Carlo BER simulation across all specified Eb/N0 values.

    For each Eb/N0 point:
      1. Transmit a fixed random bit sequence through the QPSK Tx pipeline.
      2. Corrupt the transmitted waveform with AWGN scaled to the target Eb/N0.
      3. Recover bits through the matched-filter Rx pipeline.
      4. Count bit errors and compute empirical BER.

    Parameters
    ----------
    ebn0_db_range : ndarray of float
        Eb/N0 sweep values in dB.
    n_bits : int
        Number of information bits per trial.  More bits → lower BER floor
        at the cost of longer simulation time.  Default 20 000.
    sps : int
        Samples per symbol (oversampling factor).  Default 8.
    beta : float
        RRC roll-off factor.  Default 0.35 (ETSI EN 302 307-1).
    n_taps : int
        RRC filter length (odd).  Default 101.
    seed : int
        PRNG seed for the transmitter bit generator.  Default 42.

    Returns
    -------
    ber_empirical : ndarray of float, shape (len(ebn0_db_range),)
        Empirical BER for each Eb/N0 point.

    Notes
    -----
    The transmitter seed is fixed so every Eb/N0 point is tested on the
    same bit sequence, enabling direct error-count comparison.  The channel
    AWGN uses an un-seeded RNG to provide independent noise realisations.

    The detected bit array may have a few extra trailing bits produced by
    the convolution tail; they are trimmed to n_bits before counting errors.
    """
    # Pre-compute the transmitter output once (same bits for all Eb/N0 points)
    tx_signal, tx_bits, _symbols, rrc_taps = transmit(
        n_bits=n_bits, sps=sps, beta=beta, n_taps=n_taps, seed=seed
    )

    ber_empirical = np.zeros(len(ebn0_db_range))

    for idx, ebn0_db in enumerate(ebn0_db_range):
        # --- Phase 2: AWGN Channel ---
        rx_signal = awgn_channel(
            tx_signal,
            ebn0_db=ebn0_db,
            bits_per_symbol=BITS_PER_SYMBOL,
            sps=sps,
        )

        # --- Phase 3: Receiver ---
        detected_bits = receive(rx_signal, rrc_taps, sps)

        # Trim to transmitted length (convolution tail may add extra samples)
        detected_bits = detected_bits[: n_bits]

        # --- BER Calculation ---
        n_errors = int(np.sum(tx_bits != detected_bits))
        ber = n_errors / n_bits
        ber_empirical[idx] = ber

        print(
            f"  Eb/N0 = {ebn0_db:5.1f} dB  |  "
            f"Errors: {n_errors:5d} / {n_bits}  |  "
            f"BER = {ber:.4e}"
        )

    return ber_empirical


# ---------------------------------------------------------------------------
# Waterfall Plot
# ---------------------------------------------------------------------------
def plot_ber_waterfall(
    ebn0_db_range: np.ndarray,
    ber_empirical: np.ndarray,
    ber_theory: np.ndarray,
    save_path: str = "ber_waterfall.png",
) -> None:
    """
    Plot empirical vs. theoretical BER on a logarithmic waterfall curve.

    The "waterfall" name comes from the steep descent of the BER curve as
    Eb/N0 increases beyond a threshold, resembling a waterfall on a
    semi-log plot.

    Parameters
    ----------
    ebn0_db_range : ndarray of float
        Eb/N0 axis values in dB.
    ber_empirical : ndarray of float
        Simulated BER from :func:`run_ber_simulation`.
    ber_theory : ndarray of float
        Closed-form theoretical BER from :func:`theoretical_ber`.
    save_path : str, optional
        File path to save the plot as a PNG image.  Default 'ber_waterfall.png'.

    Plot description
    ----------------
    - X-axis: Eb/N0 [dB], linear scale.
    - Y-axis: BER, logarithmic (base-10) scale.
    - Blue solid line with circle markers: empirical (simulated) BER.
    - Red dashed line with cross markers:  theoretical BER.
    - Grid: enabled for readability.
    - Y-axis floor at 1e-6 to avoid log(0) issues when BER = 0.

    The theoretical curve is:
        Pb = 0.5 · erfc(√(Eb/N0))   (QPSK, AWGN, Gray coding)

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., Figure 8.2-10.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    # Replace zero BER values with a floor so log-scale renders correctly
    ber_empirical_plot = np.maximum(ber_empirical, 1e-6)

    ax.semilogy(
        ebn0_db_range,
        ber_empirical_plot,
        "bo-",
        linewidth=1.8,
        markersize=7,
        label="Empirical BER (Monte-Carlo)",
    )
    ax.semilogy(
        ebn0_db_range,
        ber_theory,
        "r--x",
        linewidth=1.8,
        markersize=7,
        label=r"Theoretical BER: $P_b = \frac{1}{2}\,\mathrm{erfc}\!\left(\sqrt{E_b/N_0}\right)$",
    )

    ax.set_xlabel("$E_b/N_0$ (dB)", fontsize=13)
    ax.set_ylabel("Bit Error Rate (BER)", fontsize=13)
    ax.set_title(
        "QPSK BER Waterfall Curve — AWGN Channel\n"
        r"RRC Pulse Shaping ($\alpha=0.35$, ETSI EN 302 307-1 / DVB-S2)",
        fontsize=13,
    )
    ax.set_xlim(ebn0_db_range[0], ebn0_db_range[-1])
    ax.set_ylim(1e-5, 1.0)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    print(f"\nBER waterfall plot saved to: {save_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Orchestrate the full QPSK transceiver BER simulation.

    Simulation pipeline
    -------------------
    1. Define sweep parameters (Eb/N0 = 0 … 10 dB, step 1 dB).
    2. Run Monte-Carlo BER simulation via :func:`run_ber_simulation`.
    3. Compute theoretical BER via :func:`theoretical_ber`.
    4. Plot and save the waterfall curve via :func:`plot_ber_waterfall`.
    """
    print("=" * 60)
    print("  QPSK Baseband Transceiver — BER Simulation")
    print(f"  N_bits={N_BITS}, SPS={SPS}, β={BETA}, N_taps={N_TAPS}")
    print("=" * 60)

    ber_empirical = run_ber_simulation(EBN0_DB_RANGE)
    ber_theory = theoretical_ber(EBN0_DB_RANGE)

    print("\n--- Summary ---")
    print(f"{'Eb/N0 (dB)':>12}  {'BER Empirical':>15}  {'BER Theoretical':>16}")
    print("-" * 48)
    for ebn0_db, ber_e, ber_t in zip(EBN0_DB_RANGE, ber_empirical, ber_theory):
        print(f"{ebn0_db:>12.1f}  {ber_e:>15.4e}  {ber_t:>16.4e}")

    plot_ber_waterfall(EBN0_DB_RANGE, ber_empirical, ber_theory)


if __name__ == "__main__":
    main()
