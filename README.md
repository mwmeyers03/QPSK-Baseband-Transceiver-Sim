# End-to-End QPSK Baseband Transceiver Simulation



## 1. Abstract
This repository contains a pure-software, end-to-end baseband simulation of a Quadrature Phase Shift Keying (QPSK) digital communication system. It models the complete physical layer (PHY) signal chain from bit generation to symbol decision, validating empirical performance against theoretical probability models in an Additive White Gaussian Noise (AWGN) channel.

## 2. Standards Compliance & Official Documentation
This project implements pulse-shaping parameters based on digital broadcasting standards to ensure real-world applicability:
* **ETSI EN 302 307 (DVB-S2):** Utilizes standard roll-off factors ($\beta = 0.20, 0.25, 0.35$) for the Root-Raised Cosine (RRC) filters to strictly bound the spectral bandwidth and eliminate Inter-Symbol Interference (ISI).
* **ITU-R V.431-8:** Adheres to standard nomenclature for frequency and wavelength bands in RF channel modeling.

## 3. Mathematical Models & DSP Architecture

### 3.1 QPSK Modulation & Pulse Shaping (Transmitter)
Random bit streams are mapped to complex symbols in the I and Q planes. To band-limit the transmission, the upsampled symbols are convolved with a Root-Raised Cosine (RRC) filter.
The impulse response of the RRC filter is defined as:
$$h(t) = \frac{\sin\left(\pi \frac{t}{T_s}(1-\beta)\right) + 4\beta \frac{t}{T_s} \cos\left(\pi \frac{t}{T_s}(1+\beta)\right)}{\pi \frac{t}{T_s} \left(1 - \left(4\beta \frac{t}{T_s}\right)^2\right)}$$
where $T_s$ is the symbol period and $\beta$ is the roll-off factor.



### 3.2 AWGN Channel Model
The RF channel introduces Additive White Gaussian Noise. The noise variance $\sigma^2$ is strictly calculated based on the ratio of Energy per Bit ($E_b$) to Noise Power Spectral Density ($N_0$):
$$\sigma^2 = \frac{N_0}{2} = \frac{1}{2 \cdot \frac{E_b}{N_0} \cdot \log_2(M) \cdot R_s}$$
where $M$ is the modulation order ($M=4$ for QPSK) and $R_s$ is the symbol rate.

### 3.3 Matched Filtering & Demodulation (Receiver)
The received noisy signal $y(t)$ is passed through an identical RRC matched filter, downsampled at the optimal sampling instants, and mapped back to bits using maximum-likelihood decision boundaries.

### 3.4 Bit Error Rate (BER) Validation
The empirical BER is calculated by comparing the transmitted and received bitstreams and is plotted against the theoretical QPSK limit:
$$P_b = Q\left(\sqrt{\frac{2E_b}{N_0}}\right) = \frac{1}{2}\text{erfc}\left(\sqrt{\frac{E_b}{N_0}}\right)$$



## 4. Repository Architecture
├── /docs
│   └── ETSI_DVB_S2_Standard_Reference.pdf
├── /src
│   ├── main_sim.py        # Top-level simulation loop and Matplotlib plotting
│   ├── transmitter.py     # Bit generation, mapping, and RRC pulse shaping
│   ├── channel.py         # Exact AWGN noise variance calculations
│   └── receiver.py        # Matched filtering, downsampling, and detection
└── README.md

## 5. Execution Guide
1. **Prerequisites:** Python 3.10+, `numpy`, `scipy`, `matplotlib`.
2. **Run Simulation:** Execute `python src/main_sim.py`.
3. **Output:** The script sweeps $E_b/N_0$ from 0 dB to 10 dB and outputs a `ber_waterfall.png` plot, validating the DSP pipeline against the theoretical limits.

## 6. Author
**Michael W. Meyers**
*M.S. Electrical Engineering* | *IEEE Member*
