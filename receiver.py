"""
receiver.py — Phase 3: QPSK Baseband Receiver
===============================================
Implements matched filtering, optimal downsampling, maximum-likelihood (ML)
symbol detection, and Gray-coded bit de-mapping for a QPSK baseband receiver.

The receiver is the dual of the transmitter: the same RRC filter is applied
as a matched filter so that the cascade Tx-RRC → Channel → Rx-RRC achieves
the full Raised Cosine (RC) response, satisfying the Nyquist ISI-free
criterion at the sampling instants.

References
----------
- Proakis & Salehi, "Digital Communications", 5th ed., §8.2, §9.2.
- ETSI EN 302 307-1 V1.4.1 (2014-11), Annex B.
- Haykin, "Communication Systems", 4th ed., §7.4.
- Van Trees, "Detection, Estimation, and Modulation Theory", Part I, Ch. 4.
"""

import numpy as np

from transmitter import QUADRANT_TO_BITS, rrc_filter


# ---------------------------------------------------------------------------
# 1. Matched Filter
# ---------------------------------------------------------------------------
def matched_filter(rx_signal: np.ndarray, rrc_taps: np.ndarray) -> np.ndarray:
    """
    Convolve the received noisy signal with the RRC matched filter.

    The matched filter maximises the instantaneous SNR at the sampling
    instant.  For a signal corrupted by AWGN, the optimal linear receiver
    is a filter whose impulse response is the time-reversed (conjugated)
    version of the transmitted pulse.  Because the RRC pulse is real and
    symmetric, the matched filter is identical to the Tx RRC filter.

    Parameters
    ----------
    rx_signal : ndarray of complex128, shape (N,)
        Noisy received signal from the AWGN channel.
    rrc_taps : ndarray of float64, shape (n_taps,)
        RRC filter impulse response (from :func:`transmitter.rrc_filter`).
        The same coefficients are used as the Tx pulse-shaping filter,
        which is valid because the RRC is real and symmetric:
        h_matched[n] = h_RRC*[-n] = h_RRC[n].

    Returns
    -------
    mf_signal : ndarray of complex128, shape (N + n_taps - 1,)
        Output of the matched filter (full linear convolution).

    Mathematical model
    ------------------
    The optimal matched filter in AWGN has impulse response:

        h_MF(t) = h_RRC*(-t)  (time-reversed conjugate of Tx pulse)

    Because the RRC pulse is real and symmetric:
        h_MF(t) = h_RRC(t)

    The combined Tx + Rx cascade forms the full Raised Cosine:

        H_RC(f) = H_RRC(f) · H_MF(f) = |H_RRC(f)|² = H_RC(f)

    which satisfies Nyquist's first criterion for zero ISI at sampling
    instants t = k·T (Proakis & Salehi, §9.2, Theorem 9.2-1):

        h_RC(kT) = δ[k]   ⟺   Σ_n H_RC(f + n/T) = T  ∀f

    The SNR gain of the matched filter over an unmatched receiver is:

        SNR_MF / SNR_unmatched = ||h_RRC||² / N₀ = 1/N₀

    (since the RRC filter is normalised to unit energy, ||h_RRC||² = 1).

    After the two-filter cascade (each with group delay D = (n_taps-1)/2),
    the total group delay is 2·D = n_taps - 1 samples.

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., §8.2.4, Theorem 8.2-1.
    Van Trees, "Detection, Estimation, and Modulation Theory", Part I, §4.3.
    """
    mf_real = np.convolve(rx_signal.real, rrc_taps, mode="full")
    mf_imag = np.convolve(rx_signal.imag, rrc_taps, mode="full")
    return mf_real + 1j * mf_imag


# ---------------------------------------------------------------------------
# 2. Optimal Downsampling
# ---------------------------------------------------------------------------
def downsample(mf_signal: np.ndarray, sps: int, n_taps: int) -> np.ndarray:
    """
    Downsample the matched-filter output at the optimal sampling instants.

    The total group delay introduced by two cascaded RRC FIR filters (each
    of length n_taps) is n_taps - 1 samples.  The first valid symbol peak
    therefore occurs at sample index n_taps - 1.  Subsequent symbol peaks
    are spaced exactly sps samples apart.

    Parameters
    ----------
    mf_signal : ndarray of complex128, shape (M,)
        Output of the matched filter (full convolution).
    sps : int
        Samples per symbol.
    n_taps : int
        Number of taps in each RRC filter.

    Returns
    -------
    symbols_rx : ndarray of complex128, shape (N_sym,)
        Complex decision variables, one per received symbol.

    Mathematical model
    ------------------
    Each RRC FIR filter of length L has a group delay:

        D_single = (L - 1) / 2   [samples]

    Two cascaded filters introduce total group delay:

        D_total = 2 · D_single = L - 1   [samples]

    The optimal sampling index for the k-th symbol is:

        n_k = D_total + k · sps,   k = 0, 1, 2, …, N_sym - 1

    This aligns the sampler with the peak of the RC pulse response at
    each symbol period, where ISI = 0 (Nyquist criterion is satisfied).

    Any samples before index D_total or after the last valid symbol
    peak are discarded as transient edge artefacts of the convolution.

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., §8.2.4.
    Harris, "On the Use of Windows for Harmonic Analysis", IEEE Proc., 1978.
    """
    group_delay = n_taps - 1     # total delay of two cascaded RRC filters
    return mf_signal[group_delay::sps]


# ---------------------------------------------------------------------------
# 3. Maximum-Likelihood Symbol Detection (QPSK)
# ---------------------------------------------------------------------------
def detect_symbols(symbols_rx: np.ndarray) -> np.ndarray:
    """
    Apply maximum-likelihood (ML) decision boundaries to detect QPSK symbols.

    For QPSK with equal-energy, equal-prior-probability symbols corrupted by
    AWGN, the ML detector reduces to a minimum-Euclidean-distance detector,
    whose decision boundaries are the real and imaginary axes (I=0, Q=0).

    Parameters
    ----------
    symbols_rx : ndarray of complex128, shape (N_sym,)
        Noisy complex samples at the matched-filter output (decision
        variables), one per symbol period.

    Returns
    -------
    detected_bits : ndarray of int, shape (2 * N_sym,)
        Detected bit stream; two bits are recovered per symbol via the
        Gray-coded de-mapping table in :mod:`transmitter`.

    Mathematical model
    ------------------
    For AWGN channel with noise variance σ², the received sample is:

        r_k = s_k + n_k,   n_k ~ CN(0, 2σ²)

    where s_k ∈ { (±1±j)/√2 } is the transmitted symbol.

    The ML decision rule for M-ary signalling with equal priors is:

        ŝ_k = argmax_{s_m ∈ S}  p(r_k | s_m)
            = argmin_{s_m ∈ S}  |r_k − s_m|²      (minimum Euclidean distance)

    For QPSK the four constellation points occupy the four quadrants, so
    the minimum-distance decision reduces to two independent binary decisions
    on the I and Q components:

        I-decision:  b_0 = 0 if Re{r_k} ≥ 0, else b_0 = 1
        Q-decision:  b_1 = 0 if Im{r_k} ≥ 0, else b_1 = 1

    These two sign decisions reproduce the Gray-code de-mapping:

        Quadrant (sign(I), sign(Q)) → bit-pair:
            (+1, +1) → (0, 0)
            (-1, +1) → (0, 1)
            (-1, -1) → (1, 1)
            (+1, -1) → (1, 0)

    The decision boundaries are the two axes, which are equidistant from
    each pair of neighbouring constellation points, confirming ML optimality
    under symmetric AWGN.

    References
    ----------
    Proakis & Salehi, "Digital Communications", 5th ed., §4.2, §8.2.5.
    Van Trees, "Detection, Estimation, and Modulation Theory", Part I, §4.4.
    """
    bits_out = []
    for sample in symbols_rx:
        i_sign = 1 if sample.real >= 0 else -1
        q_sign = 1 if sample.imag >= 0 else -1
        bit_pair = QUADRANT_TO_BITS[(i_sign, q_sign)]
        bits_out.extend(bit_pair)
    return np.array(bits_out, dtype=int)


# ---------------------------------------------------------------------------
# Top-level receiver function
# ---------------------------------------------------------------------------
def receive(
    rx_signal: np.ndarray,
    rrc_taps: np.ndarray,
    sps: int,
) -> np.ndarray:
    """
    End-to-end receiver pipeline.

    Applies matched filtering → optimal downsampling → ML symbol detection
    → Gray-coded bit de-mapping.

    Parameters
    ----------
    rx_signal : ndarray of complex128
        Noisy baseband signal from the AWGN channel.
    rrc_taps : ndarray of float64
        RRC filter coefficients (identical to those used at the transmitter).
    sps : int
        Samples per symbol.

    Returns
    -------
    detected_bits : ndarray of int
        Recovered bit stream.

    Notes
    -----
    The number of detected bits may differ from the number of transmitted
    bits by a small amount due to tail samples produced by the convolution
    transient.  The caller (main_sim.py) is responsible for trimming to the
    known transmitted length before computing the BER.
    """
    n_taps = len(rrc_taps)
    mf_out = matched_filter(rx_signal, rrc_taps)
    symbols_rx = downsample(mf_out, sps, n_taps)
    detected_bits = detect_symbols(symbols_rx)
    return detected_bits
