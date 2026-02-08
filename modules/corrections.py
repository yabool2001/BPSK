from modules import modulation , sdr
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.signal import correlate , butter, lfilter
from numba import jit
import time as t
import tomllib

# Wczytaj plik TOML z konfiguracją
with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

def full_compensation_v0_0_0 ( samples , preamble_samples ) :
    """
    Improved full compensation pipeline:
      1) NEW FEATURE: coarse CFO estimate from preamble (apply frequency rotation) -> multiplicative correction
      2) PLL (fine tracking)
      3) phase offset correction via correlation with preamble
      4) IQ balance

    Returns corrected samples (complex numpy array).
    """
    fs = sdr.F_S
    sps = modulation.SPS

    # 1) Coarse CFO estimate and correction
    ts = t.perf_counter_ns ()
    coarse_f = estimate_cfo_from_preamble_v0_0_0 ( samples , preamble_samples , fs, sps )
    if toml_settings[ "log" ][ "verbose_2" ] : print (f"estimate_cfo_from_preamble_v0_0_0 w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")
    # Apply coarse correction
    if coarse_f != 0.0 :
    # Alternative apply coarse correction: if abs(coarse_f) > 1e-12:
        n = np.arange ( len ( samples ) )
        samples = samples * np.exp ( -1j * 2.0 * np.pi * coarse_f * n / fs )

    # 2) PLL-based fine tracking
    ts = t.perf_counter_ns ()
    pl_corrected = pll_v0_0_0 ( samples , freq_offset_initial = 0.0 )
    if toml_settings[ "log" ][ "verbose_2" ] : print (f"pll_v0_0_0 w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")

    # 3) Phase offset correction using preamble
    ts = t.perf_counter_ns ()
    rx_phase_corrected = correct_phase_offset_v0_0_0 ( pl_corrected , preamble_samples )
    if toml_settings[ "log" ][ "verbose_2" ] : print (f"correct_phase_offset_v3 w czasie [ms]: {( t.perf_counter_ns () - ts ) / 1e6:.1f} ")

    # 4) IQ imbalance compensation
    rx_final_corrected = iq_balance_v0_0_0 ( rx_phase_corrected )

    return rx_final_corrected

def estimate_cfo_from_preamble_v0_0_0 ( rx , preamble , fs , sps ) :
    """
    Simple coarse CFO estimator using a known preamble.
    Uses products of samples separated by `sps` (M) and returns frequency offset in Hz.
    """
    if rx is None or preamble is None:
        return 0.0
    corr = np.correlate ( rx , preamble , mode = 'valid' )
    if corr.size == 0:
        return 0.0
    peak = np.argmax ( np.abs ( corr ) )
    seg_len = len ( preamble )
    # take segment aligned to preamble (clip if necessary)
    if peak + seg_len <= len ( rx ):
        seg = rx[ peak : peak + seg_len ]
    else:
        seg = rx[ peak : ]
    if len ( seg ) <= sps :
        return 0.0
    prods = seg[ sps : ] * np.conj ( seg[ : -sps ] )
    # average product to reduce noise
    avg = np.mean ( prods )
    delta = np.angle ( avg )
    f_offset = delta * fs / (2.0 * np.pi * sps)
    return float ( f_offset )

def pll_v0_0_0 ( rx_samples , freq_offset_initial ) :

    loop_bw = 2 * np.pi * 100 / sdr.F_S  # szerokość pasma pętli (np. 50 Hz)
    alpha = loop_bw
    beta = alpha**2 / 4
    
    # Wywołanie skompilowanego kernela Numby
    return pll_kernel_numba ( rx_samples , freq_offset_initial , alpha , beta )

def correct_phase_offset_v0_0_0 ( samples , preamble_samples ) :
    # Korelacja bez sprzężenia (dla detekcji offsetu fazowego)
    correlation = np.correlate ( samples , preamble_samples , mode='valid' )
    max_corr = correlation[ np.argmax ( np.abs ( correlation ) ) ]
    phase_offset = np.angle ( max_corr )

    # Korekcja rotacji fazowej
    rx_corrected = samples * np.exp ( -1j * phase_offset )

    # Detekcja lustrzanego odbicia (poprzez porównanie energii korelacji dla oryginalnej i sprzężonej preambuły)
    corr_normal = np.max ( np.abs ( np.correlate ( rx_corrected , preamble_samples , mode = 'valid' ) ) )
    corr_conj = np.max ( np.abs ( np.correlate ( rx_corrected , np.conj ( preamble_samples ) , mode = 'valid' ) ) )

    if corr_conj > corr_normal:
        rx_corrected = np.conj(rx_corrected)

    return rx_corrected

def iq_balance_v0_0_0 ( samples ) :
    I = np.real ( samples )
    Q = np.imag ( samples )
    Q -= np.mean ( Q )
    scale = np.std ( I ) / np.std ( Q )
    Q *= scale
    return I + 1j * Q

@jit(nopython=True)
def pll_kernel_numba ( rx_samples , freq_estimate , alpha , beta ) :

    phase_estimate = 0.0
    corrected_samples = np.zeros_like ( rx_samples )

    for n in range ( len ( rx_samples ) ) :
        sample = rx_samples[ n ]
        # Korekcja aktualną estymacją
        val = sample * np.exp ( -1j * phase_estimate )
        corrected_samples[ n ] = val

        # Błąd fazy z demodulowanego symbolu BPSK
        # np.real zwraca float, więc np.sign działa poprawnie w Numbie
        error = np.sign ( np.real ( val ) ) * np.imag ( val )

        # Aktualizacja estymacji częstotliwości i fazy
        freq_estimate += beta * error
        phase_estimate += freq_estimate + alpha * error

    return corrected_samples
