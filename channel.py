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
        It is assumed to have unit average symbol energy (Es = 1) after
        matched filtering; the RRC normalisation in transmitter.py ensures
        this.
    ebn0_db : float
        Eb/N0 in decibels (dB).  This is the ratio of energy per
        information bit (Eb) to one-sided noise power spectral density (N0).
    bits_per_symbol : int, optional
        Number of information bits per modulation symbol.
        QPSK: k = log2(M) = log2(4) = 2.  Default 2.
    sps : int, optional
        Samples per symbol (oversampling factor).  Default 8.

    Returns
    -------
    rx_signal : ndarray of complex128, shape (N,)
        Received signal with complex AWGN added.

    Mathematical model
    ------------------
    Given:
        Eb/N0 [linear] = 10^(Eb/N0_dB / 10)

    The symbol energy is:
        Es = k · Eb           (k = bits per symbol = 2 for QPSK)

    For a complex baseband signal sampled at rate fs = sps/T (sps samples
    per symbol period T), the noise PSD N0 is spread across all samples.
    The noise variance per real/imaginary component is:

        σ² = N0 / 2 = Es / (2 · k · Eb/N0)
           = sps / (2 · k · Eb/N0_linear)

    Because the RRC filter is normalised to unit energy (||h||² = 1) and
    the symbols have unit energy (Es = 1), the average power of the
    pulse-shaped signal per sample is Es/sps = 1/sps.  The noise must be
    scaled accordingly:

        σ² = (Es / sps) / (2 · k · Eb/N0_linear)
           = 1 / (2 · k · sps · Eb/N0_linear)

    where:
        · The factor of 2 in the denominator accounts for the two-sided
          noise (real + imaginary components each carry σ² variance).
        · The factor k = bits_per_symbol accounts for the energy mapping
          Eb → Es.
        · The factor sps normalises for oversampling; each symbol is
          represented by sps samples, diluting the signal power per sample.

    Complex AWGN is generated as:
        n[i] = σ · (n_I[i] + j·n_Q[i]),   n_I, n_Q ~ N(0, 1)

    where n_I and n_Q are independent real Gaussian random variables so
    that E[|n|²] = 2σ².

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., §4.1, Eq. (4.1-9).
    Sklar, "Digital Communications", 2nd ed., §4.2.
    """
    ebn0_linear = 10.0 ** (ebn0_db / 10.0)

    # Noise variance per real or imaginary component
    sigma_sq = 1.0 / (2.0 * bits_per_symbol * sps * ebn0_linear)
    sigma = np.sqrt(sigma_sq)

    # Generate complex AWGN: real and imaginary parts are independent N(0, σ²)
    rng = np.random.default_rng()          # non-seeded for Monte-Carlo variety
    noise = sigma * (
        rng.standard_normal(len(tx_signal))
        + 1j * rng.standard_normal(len(tx_signal))
    )

    return tx_signal + noise
