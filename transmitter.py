"""
transmitter.py — Phase 1: QPSK Baseband Transmitter
=====================================================
Implements bit generation, Gray-coded QPSK symbol mapping, upsampling, and
Root-Raised Cosine (RRC) pulse-shaping as specified in ETSI EN 302 307-1
(DVB-S2), Annex B.

References
----------
- ETSI EN 302 307-1 V1.4.1 (2014-11), Section 5.4.3 / Annex B:
  Roll-off factor α = 0.35 mandated for DVB-S2 carriers.
- Proakis & Salehi, "Digital Communications", 5th ed., Ch. 8, §8.2.
- Haykin, "Communication Systems", 4th ed., Ch. 7.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRAY_QPSK_MAP = {
    (0, 0): complex( 1,  1) / np.sqrt(2),   # I=+1, Q=+1  (Gray: 00 → π/4)
    (0, 1): complex(-1,  1) / np.sqrt(2),   # I=-1, Q=+1  (Gray: 01 → 3π/4)
    (1, 1): complex(-1, -1) / np.sqrt(2),   # I=-1, Q=-1  (Gray: 11 → 5π/4)
    (1, 0): complex( 1, -1) / np.sqrt(2),   # I=+1, Q=-1  (Gray: 10 → 7π/4)
}
"""
Gray-coded QPSK constellation map (unit-energy symbols).

Bit-pair assignment follows Gray coding so that adjacent symbols differ by
exactly one bit, minimising BER under high-SNR conditions.

Constellation points lie on the unit circle at ±45°, ±135°:
  s_k = (1/√2) · (±1 ± j)

The 1/√2 normalisation ensures average symbol energy Es = 1.
"""

GRAY_QPSK_DEMAP = {v: k for k, v in GRAY_QPSK_MAP.items()}

# Quadrant → bit-pair lookup used during ML detection
QUADRANT_TO_BITS = {
    ( 1,  1): (0, 0),
    (-1,  1): (0, 1),
    (-1, -1): (1, 1),
    ( 1, -1): (1, 0),
}


# ---------------------------------------------------------------------------
# 1. Bit Generation
# ---------------------------------------------------------------------------
def generate_bits(n_bits: int, seed: int = 42) -> np.ndarray:
    """
    Generate a pseudorandom binary sequence (PRBS).

    Parameters
    ----------
    n_bits : int
        Total number of bits to generate.  Must be even so the sequence can
        be packed into QPSK di-bits (2 bits per symbol).
    seed : int, optional
        NumPy random seed for reproducibility.  Default 42.

    Returns
    -------
    bits : ndarray of int, shape (n_bits,)
        Array of i.i.d. Bernoulli(0.5) samples ∈ {0, 1}.

    Mathematical model
    ------------------
    Each bit b_k ~ Bernoulli(p=0.5), independent of all others.
    The expected bit energy Eb is absorbed into the SNR definition used in
    channel.py; the raw bit values here are dimensionless {0,1}.
    """
    if n_bits % 2 != 0:
        raise ValueError(f"n_bits must be even for QPSK; got {n_bits}.")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=n_bits, dtype=int)


# ---------------------------------------------------------------------------
# 2. QPSK Symbol Mapping (Gray coding)
# ---------------------------------------------------------------------------
def map_bits_to_symbols(bits: np.ndarray) -> np.ndarray:
    """
    Map binary data to Gray-coded QPSK complex symbols.

    Two consecutive bits (di-bit) are mapped to one of four constellation
    points.  Gray coding is applied so adjacent symbols differ by exactly
    one bit.

    Parameters
    ----------
    bits : ndarray of int, shape (N,)
        Input bit stream.  N must be even.

    Returns
    -------
    symbols : ndarray of complex128, shape (N//2,)
        Unit-energy QPSK symbols with Es = 1.

    Mathematical model
    ------------------
    For QPSK the symbol alphabet is:

        S = { e^{j·π/4}, e^{j·3π/4}, e^{j·5π/4}, e^{j·7π/4} }
          = { (+1+j)/√2, (-1+j)/√2, (-1-j)/√2, (+1-j)/√2 }

    Gray-code mapping (MSB, LSB) → symbol:
        00 → (+1+j)/√2
        01 → (-1+j)/√2
        11 → (-1-j)/√2
        10 → (+1-j)/√2

    This mapping minimises BER because the two nearest-neighbour symbols
    differ by exactly one bit.

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., Table 4-3.
    """
    if len(bits) % 2 != 0:
        raise ValueError("Bit array length must be even for QPSK mapping.")

    dibits = bits.reshape(-1, 2)
    symbols = np.array(
        [GRAY_QPSK_MAP[(int(b[0]), int(b[1]))] for b in dibits],
        dtype=complex,
    )
    return symbols


# ---------------------------------------------------------------------------
# 3. Upsampling
# ---------------------------------------------------------------------------
def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """
    Upsample a symbol sequence by inserting (sps-1) zeros between samples.

    Parameters
    ----------
    symbols : ndarray of complex128, shape (N_sym,)
        Baseband symbol sequence.
    sps : int
        Samples per symbol (oversampling factor), e.g. 8.

    Returns
    -------
    upsampled : ndarray of complex128, shape (N_sym * sps,)
        Zero-padded (up-sampled) symbol stream.

    Mathematical model
    ------------------
    The upsampled sequence x[n] is defined as:

        x[n] = s[n/sps],   if n is a multiple of sps
               0,           otherwise

    where s[k] is the k-th symbol.  Inserting zeros in the time domain
    creates spectral images spaced at fs/sps in the frequency domain.
    The subsequent RRC filter removes these images while shaping the pulse.
    This is equivalent to ideal D/A conversion at the higher sample rate
    followed by pulse shaping.

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., §8.2.2.
    """
    n_sym = len(symbols)
    upsampled = np.zeros(n_sym * sps, dtype=complex)
    upsampled[::sps] = symbols
    return upsampled


# ---------------------------------------------------------------------------
# 4. Root-Raised Cosine Filter
# ---------------------------------------------------------------------------
def rrc_filter(beta: float, sps: int, n_taps: int) -> np.ndarray:
    """
    Compute the impulse response of a Root-Raised Cosine (RRC) FIR filter.

    The RRC filter is specified in ETSI EN 302 307-1 (DVB-S2), Annex B as
    the mandatory pulse-shaping filter with roll-off factor α = 0.35.

    Parameters
    ----------
    beta : float
        Roll-off factor α ∈ (0, 1].  ETSI EN 302 307-1 mandates α = 0.35.
    sps : int
        Samples per symbol (oversampling factor).
    n_taps : int
        Total number of filter taps.  Should be odd so the filter is
        symmetric and has integer group delay.

    Returns
    -------
    h : ndarray of float64, shape (n_taps,)
        Normalised RRC filter coefficients (unit energy: ||h||² = 1).

    Mathematical model
    ------------------
    The Raised Cosine (RC) spectrum is:

        H_RC(f) = T,                           |f| ≤ (1-α)/(2T)
                  T/2·[1 + cos(πT/α·(|f| - (1-α)/(2T)))],
                                               (1-α)/(2T) < |f| ≤ (1+α)/(2T)
                  0,                           |f| > (1+α)/(2T)

    The RRC is the square root in the frequency domain: H_RRC(f) = √H_RC(f),
    so that the combined Tx RRC + Rx matched RRC = RC, which achieves
    zero inter-symbol interference (ISI) at the Nyquist sampling instants
    (Nyquist's first criterion).

    The corresponding discrete-time impulse response is:

        h[n] = (4α/π·√T) · [cos((1+α)π·n/T) + sin((1-α)π·n/T)/(4α·n/T)]
               ─────────────────────────────────────────────────────────────
               1 - (4α·n/T)²

    where the continuous-time symbol period T is represented in discrete
    samples as sps (i.e., T ≡ sps), so 1/√T = 1/√sps in the code.
    The time index t = n/sps converts sample index to symbol periods.
    The special cases at t = 0 and t = ±1/(4α) (i.e. n = ±sps/(4α))
    are evaluated via L'Hôpital's rule:

        h[0]            = (1/√sps) · (1 - α + 4α/π)
        h[±sps/(4α)]    = (α/√(2·sps)) · [(1+2/π)·sin(π/(4α))
                                           + (1-2/π)·cos(π/(4α))]

    The filter is normalised so that ||h||² = 1 (unit energy), ensuring
    the matched-filter combiner has unit gain at the decision point.

    References
    ----------
    ETSI EN 302 307-1 V1.4.1 (2014-11), Annex B — Roll-off factor α = 0.35.
    Proakis & Salehi, "Digital Communications", 5th ed., §9.2, Eq. (9.2-23).
    Harris, "On the Use of Windows for Harmonic Analysis", IEEE Proc., 1978.
    """
    if n_taps % 2 == 0:
        raise ValueError("n_taps must be odd for a symmetric linear-phase FIR filter.")

    half = (n_taps - 1) // 2
    t = np.arange(-half, half + 1, dtype=float) / sps   # time in symbol periods

    h = np.zeros(n_taps, dtype=float)

    for i, ti in enumerate(t):
        if ti == 0.0:
            # L'Hôpital case 1: t = 0
            h[i] = (1.0 / np.sqrt(sps)) * (1.0 - beta + 4.0 * beta / np.pi)
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
            # L'Hôpital case 2: t = ±1/(4β)
            h[i] = (beta / np.sqrt(2.0 * sps)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            numerator = (
                np.sin(np.pi * ti * (1.0 - beta))
                + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            )
            denominator = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = (1.0 / np.sqrt(sps)) * numerator / denominator

    # Normalise to unit energy
    h /= np.sqrt(np.sum(h ** 2))
    return h


# ---------------------------------------------------------------------------
# 5. Pulse-shape (Tx RRC filter application)
# ---------------------------------------------------------------------------
def pulse_shape(upsampled_signal: np.ndarray, rrc_taps: np.ndarray) -> np.ndarray:
    """
    Convolve the upsampled symbol stream with the RRC pulse-shaping filter.

    Parameters
    ----------
    upsampled_signal : ndarray of complex128, shape (N,)
        Zero-padded upsampled symbol stream from :func:`upsample`.
    rrc_taps : ndarray of float64, shape (n_taps,)
        RRC filter impulse response from :func:`rrc_filter`.

    Returns
    -------
    tx_signal : ndarray of complex128, shape (N + n_taps - 1,)
        Pulse-shaped baseband transmit waveform (full convolution).

    Mathematical model
    ------------------
    The transmitted baseband signal is:

        s(t) = Σ_k  a_k · h_RRC(t − k·T)

    where a_k is the k-th QPSK symbol and h_RRC is the RRC pulse.
    In discrete time, this is a linear convolution:

        s[n] = x[n] * h[n]  (linear convolution, not circular)

    Because linear convolution is used, the output length is
    N_in + N_taps − 1.  The first and last (N_taps−1)/2 samples contain
    transient edge effects (group delay artefacts); these are compensated
    at the receiver during downsampling.

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., §8.2.3.
    """
    # Real and imaginary parts are filtered separately to avoid any complex
    # precision issues, then recombined.
    tx_real = np.convolve(upsampled_signal.real, rrc_taps, mode="full")
    tx_imag = np.convolve(upsampled_signal.imag, rrc_taps, mode="full")
    return tx_real + 1j * tx_imag


# ---------------------------------------------------------------------------
# Top-level transmitter function
# ---------------------------------------------------------------------------
def transmit(
    n_bits: int = 4096,
    sps: int = 8,
    beta: float = 0.35,
    n_taps: int = 101,
    seed: int = 42,
):
    """
    End-to-end transmitter pipeline.

    Generates bits → maps to QPSK symbols → upsamples → RRC pulse-shapes.

    Parameters
    ----------
    n_bits : int
        Number of information bits.  Default 4096.
    sps : int
        Samples per symbol.  Default 8.
    beta : float
        RRC roll-off factor (ETSI EN 302 307-1 mandates 0.35).  Default 0.35.
    n_taps : int
        Number of RRC filter taps.  Default 101 (≈ 12.5 symbol periods).
    seed : int
        PRNG seed for bit generation.  Default 42.

    Returns
    -------
    tx_signal : ndarray of complex128
        Pulse-shaped transmitted baseband waveform.
    bits : ndarray of int
        Original transmitted bit sequence.
    symbols : ndarray of complex128
        QPSK symbol sequence (before upsampling).
    rrc_taps : ndarray of float64
        RRC filter coefficients.
    """
    bits = generate_bits(n_bits, seed=seed)
    symbols = map_bits_to_symbols(bits)
    upsampled = upsample(symbols, sps)
    rrc_taps = rrc_filter(beta, sps, n_taps)
    tx_signal = pulse_shape(upsampled, rrc_taps)
    return tx_signal, bits, symbols, rrc_taps
