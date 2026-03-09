"""
channel.py — Phase 2: AWGN Channel Model
=========================================
Adds complex Additive White Gaussian Noise (AWGN) to the transmitted
baseband waveform.  The noise variance is derived exactly from the
specified Eb/N0 (energy-per-bit to noise power spectral density ratio),
the modulation order, and the oversampling factor.

References
----------
- Proakis & Salehi, "Digital Communications", 5th ed., §4.1 & §8.1.
- Sklar, "Digital Communications: Fundamentals and Applications", 2nd ed.,
  Ch. 4.
- ITU-R S.521-6, "Calculation of the maximum permissible level of
  interference in a satellite network."
"""

import numpy as np


def awgn_channel(
    tx_signal: np.ndarray,
    ebn0_db: float,
    bits_per_symbol: int = 2,
    sps: int = 8,
) -> np.ndarray:
    """
    Pass the transmitted signal through an AWGN channel.

    Parameters
    ----------
    tx_signal : ndarray of complex128, shape (N,)
        Pulse-shaped baseband transmit waveform from the transmitter.
        The RRC normalisation in transmitter.py ensures unit discrete-time
        filter energy (||h_RRC||² = 1), so the symbol energy at the
        matched-filter decision point equals Es = 1.
    ebn0_db : float
        Eb/N0 in decibels (dB).  This is the ratio of energy per
        information bit (Eb) to one-sided noise power spectral density (N0).
    bits_per_symbol : int, optional
        Number of information bits per modulation symbol.
        QPSK: k = log2(M) = log2(4) = 2.  Default 2.
    sps : int, optional
        Samples per symbol (oversampling factor).  Retained as a parameter
        for API consistency; it does NOT appear in the noise-variance formula
        (see derivation below).  Default 8.

    Returns
    -------
    rx_signal : ndarray of complex128, shape (N,)
        Received signal with complex AWGN added.

    Mathematical model
    ------------------
    Given:
        Eb/N0 [linear] = 10^(Eb/N0_dB / 10)

    For unit-energy QPSK symbols (Es = 1) with k bits per symbol:
        Eb = Es / k = 1 / k

    The one-sided noise PSD:
        N0 = Eb / (Eb/N0_linear) = 1 / (k · Eb/N0_linear)

    In the discrete-time matched-filter (MF) framework, the MF is the
    same RRC filter applied at the receiver (identical to the Tx filter).
    Because the RRC filter has unit discrete-time energy:
        ||h_RRC||²_discrete = Σ_n h²[n] = 1

    the MF output noise variance (per real or imaginary component) at the
    optimal sampling instant equals the input noise variance per sample:
        σ²_MF = σ²_input · ||h_MF||²_discrete = σ²_input · 1

    For the empirical BER to equal the theoretical QPSK BER:
        Pb = 0.5 · erfc(√(Eb/N0))

    the decision variable must satisfy (per QPSK arm, where A = 1/√2 is
    the per-arm signal amplitude for unit-energy QPSK symbols):
        Q(A / σ_MF) = Q(√(2·Eb/N0))
        σ²_MF = A² / (2·Eb/N0) = (1/2) / (2·Eb/N0) = 1 / (4·Eb/N0)

    For k = 2 bits per symbol with unit-energy symbols Es = 1, Eb = Es/k = 1/2,
    and σ²_MF = σ²_input (from the unit-energy MF identity above):

        σ² = N0 / 2 = 1 / (2 · k · Eb/N0_linear)

    Key point: there is no sps factor here.  The discrete-time MF collects
    the signal energy spread over sps samples through convolution, providing
    the full matched-filter processing gain via the Σ_n h²[n] = 1 identity
    regardless of the oversampling rate.

    Complex AWGN is generated as:
        n[i] = σ · (n_I[i] + j·n_Q[i]),   n_I, n_Q ~ N(0, 1)

    so that E[|n[i]|²] = 2σ² (both components combined).

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., §4.1, §8.2.4,
        Eq. (4.1-9) and Theorem 8.2-1.
    Sklar, "Digital Communications", 2nd ed., §4.2, §6.2.
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    # Noise variance per real or imaginary component: σ² = N0/2 = 1/(2·k·Eb/N0)
    sigma_sq = 1.0 / (2.0 * bits_per_symbol * ebn0_linear)
    sigma = np.sqrt(sigma_sq)

    # Generate complex AWGN: real and imaginary parts are independent N(0, σ²)
    rng = np.random.default_rng()          # non-seeded for Monte-Carlo variety
    noise = sigma * (
        rng.standard_normal(len(tx_signal))
        + 1j * rng.standard_normal(len(tx_signal))
    )

    return tx_signal + noise
