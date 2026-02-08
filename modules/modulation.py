import numpy as np
import tomllib

from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from modules import filters , packet , plot
from scipy.signal import upfirdn , correlate

with open ( "settings.toml" , "rb" ) as settings_file :
    toml_settings = tomllib.load ( settings_file )

SPS  = int ( toml_settings[ "bpsk" ][ "SPS" ] )

def bpsk_symbols_2_bits_v0_0_0 ( symbols : NDArray[ np.float64 ] ) -> NDArray[ np.uint8 ] :
    """
    Odwrotna funkcja do create_bpsk_symbols_v0_0_0_fastest.
    Mapuje symbole BPSK (-1+0j -> 0, 1+0j -> 1) na bity.
    """
    return ( symbols > 0 ).astype ( np.uint8 )

def create_bpsk_symbols_v0_0_0_fastest_short ( bits : NDArray[ np.uint8 ] ) -> NDArray[ np.complex128 ] :
    return np.require ( bits , np.uint8 , ['C'] ) * 2.0 - 1.0 + 0j

def create_bpsk_symbols_v0_0_0_fastest ( bits : NDArray[ np.uint8 ] ) -> NDArray[ np.complex128 ] :
    bits = np.require ( bits , dtype = np.uint8 , requirements = [ 'C' ] )  # gwarantuje uint8 + contiguous
    return ( bits * 2.0 - 1.0 ).astype ( np.complex128 )

def zero_quadrature ( samples : NDArray[ np.complex128 ] ) -> NDArray[ np.complex128 ] :
    """
    Zeruje składową Q (urojoną) sygnału zespolonego, pozostawiając tylko składową I (rzeczywistą).
    """
    return np.real ( samples ) + 0j

def generate_barker13_bpsk_samples_v0_0_0 ( clipped = False ) -> NDArray[ np.complex128 ] :
    symbols = create_bpsk_symbols ( packet.BARKER13_BITS )
    samples = filters.apply_tx_rrc_filter_v0_1_3 ( symbols , True )
    if clipped :
        tail_length = ( filters.SPAN - 1 ) // 2 * SPS  # Oblicz ogon filtra RRC
        samples = samples[ : -tail_length ]  # Przytnij ogon na końcu
    return samples




















def cw ( buffer_size , scale: str ) -> np.complex128 :

    # ADALM-Pluto full scale for 16-bit DAC 2^15 - 1
    # ADALM-Pluto secure scale 2^14 
    iq = np.ones ( buffer_size ) * scale + 0j  # Stała wartość kompleksowa (DC na I, Q=0)
    return iq

def bpsk_modulation ( bpsk_symbols ) :
    zeros = np.zeros_like ( bpsk_symbols )
    zeros[bpsk_symbols == -1] = 180
    x_radians = zeros*np.pi/180.0 # sin() and cos() takes in radians
    samples = np.cos(x_radians) + 1j*0 # this produces our QPSK complex symbols
    #samples = np.repeat(symbols, 4) # 4 samples per symbol (rectangular pulses) ale to robi rrc
    samples *= 2**14 # The PlutoSDR expects samples to be between -2^14 and +2^14, not -1 and +1 like some SDRs
    # plot_tx_waveform ( samples )
    pass

def create_bpsk_symbols_v0_1_6 ( bits : NDArray[ np.uint8 ] ) -> NDArray[ np.complex128 ] :
    # Map 0 -> -1, 1 -> +1
    return np.where ( bits == 1 , 1.0 + 0j , -1.0 + 0j ).astype ( np.complex128 )


def create_bpsk_symbols_v0_1_5 ( bits ) -> np.complex128 :
    # Map 0 -> -1, 1 -> +1
    symbols_real = np.where ( bits == 1 , 1.0 , -1.0 ).astype ( np.float64 )
    # return complex symbols (Q=0)
    return ( symbols_real + 0j ).astype( np.complex128 )

def create_bpsk_symbols ( bits ) :
    return np.array ( [ 1.0 if bit else -1.0 for bit in bits ] , dtype = np.int64 )

def modulate_bpsk ( bits , sps = 4 , beta = 0.35 , span = 11 ) :
    symbols = create_bpsk_symbols ( bits )
    rrc = filters.rrc_filter_v4 ( sps , beta , span )
    shaped = upfirdn ( rrc , symbols , up = sps )
    return ( shaped + 0j ).astype ( np.complex128 )

def upsample_symbols ( symbols: np.ndarray , sps: int ) -> np.ndarray :
    """
    Zwraca ciąg zinterpolowany przez zero-stuffing (impulse upsampling).
    """
    upsampled = np.zeros ( len ( symbols ) * sps , dtype = symbols.dtype )
    upsampled[ ::sps ] = symbols
    return upsampled

def bpsk_symbols_2_bits ( symbols ) :
    return ( symbols.real > 0 ).astype ( int )

def signal_correlation(samples, lag=1):
    """
    Detekcja obecności sygnału na podstawie korelacji między sąsiednimi próbkami.

    samples  : kompleksowe próbki z RX (np. z PyADI)
    lag      : opóźnienie w próbkach (1 = sąsiednie)

    Zwraca: (float) wartość korelacji (0–1 przy szumie, wyżej przy sygnale)
    """
    if len(samples) <= lag:
        return 0.0

    x = samples[:-lag]
    y = samples[lag:]

    corr = np.vdot(x, y)
    norm = np.sqrt(np.vdot(x, x).real * np.vdot(y, y).real)

    corr_norm = np.abs(corr) / (norm + 1e-12)
    return corr_norm

def get_barker13_bpsk_samples_v0_1_3 ( clipped = False ) -> NDArray[ np.complex128 ] :
    symbols = create_bpsk_symbols ( ops_packet.BARKER13_BITS )
    samples = filters.apply_tx_rrc_filter_v0_1_3 ( symbols , True )
    if clipped :
        samples = samples[ :72 ] #Uwaga to tylko chyba dobrze działa dla SPS=4
        #samples = samples[ 18:72 ]
    #plot.plot_complex_waveform ( samples , "  barker13 samples")
    return samples

def get_barker13_bpsk_samples ( sps , rrc_beta , rrc_span , clipped = False ) :
    symbols = create_bpsk_symbols ( ops_packet.BARKER13_BITS )
    samples = filters.apply_tx_rrc_filter ( symbols , sps , rrc_beta , rrc_span , True )
    if clipped :
        samples = samples[ :72 ]
        #samples = samples[ 18:72 ]
    #plot.plot_complex_waveform ( samples , "  barker13 samples")
    return samples

def normalized_cross_correlation ( signal , template ) :
    template = (template - np.mean(template)) / np.std(template)
    n = len(template)
    corr = []

    for i in range(len(signal) - n + 1):
        window = signal[i:i+n]
        if np.std(window) == 0:  # unikanie dzielenia przez 0
            corr.append(0)
            continue
        window_norm = (window - np.mean(window)) / np.std(window)
        corr.append(np.sum(window_norm * template))

    return np.array(corr)

def fast_normalized_cross_correlation ( signal , template ) :
    template = (template - np.mean(template)) / np.std(template)
    n = len(template)

    # Użycie sliding_window_view do uzyskania wszystkich okien naraz
    windows = sliding_window_view(signal, n)  # shape: (len(signal)-n+1, n)

    # Normalizacja każdego okna
    windows_mean = np.mean(windows, axis=1, keepdims=True)
    windows_std = np.std(windows, axis=1, keepdims=True)
    windows_std[windows_std == 0] = 1  # unikanie dzielenia przez zero

    windows_norm = (windows - windows_mean) / windows_std

    # Korelacja przez iloczyn skalarny każdego okna z template
    corr = np.dot(windows_norm, template)

    return corr


import numpy as np

def fft_normalized_cross_correlation(signal, template):
    """
    Superszybka znormalizowana korelacja w trybie 'full' z wykorzystaniem FFT i rolling sum.
    Zakłada sygnał i template jako realne 1D tablice.

    Zwraca:
    - norm_corr: korelacja znormalizowana (float64), długość = len(signal) + len(template) - 1
    """
    signal = np.real(np.asarray(signal, dtype=np.float64))
    template = np.real(np.asarray(template, dtype=np.float64))
    n = len(template)
    m = len(signal)

    # --- 1. Normalizacja szablonu
    template = (template - np.mean(template)) / np.std(template)

    # --- 2. Cross-korelacja przez FFT (szablon musi być odwrócony!)
    size = m + n - 1
    fft_size = 1 << (size - 1).bit_length()  # najbliższa potęga 2 dla FFT

    template_padded = np.zeros(fft_size)
    signal_padded = np.zeros(fft_size)
    template_padded[:n] = template[::-1]
    signal_padded[:m] = signal

    fft_template = np.fft.fft(template_padded)
    fft_signal = np.fft.fft(signal_padded)

    corr = np.fft.ifft(fft_template * fft_signal).real[:size]

    # --- 3. Rolling mean and std of signal (dla okien przesuwanych)
    # rolling sum and sum of squares
    signal_sq = signal ** 2
    cumsum = np.zeros(m + 1)
    cumsum2 = np.zeros(m + 1)
    cumsum[1:] = np.cumsum(signal)
    cumsum2[1:] = np.cumsum(signal_sq)

    window_sum = cumsum[n:] - cumsum[:-n]
    window_sum2 = cumsum2[n:] - cumsum2[:-n]

    mean = window_sum / n
    std = np.sqrt(window_sum2 / n - mean**2)
    std[std == 0] = 1

    # --- 4. Dociągnij std do rozmiaru korelacji ('full')
    std_full = np.pad(std, (n - 1, size - len(std) - (n - 1)), constant_values=1)

    # --- 5. Znormalizowana korelacja
    norm_corr = corr / (std_full * n)

    return norm_corr


def group_peaks_by_distance(peaks, corr, min_distance=3):
    """
    Redukuje listę peaków do pojedynczych wartości w każdej grupie sąsiadujących indeksów.
    
    peaks: np.array z indeksami wykryć
    corr: np.array z wartościami korelacji w tych punktach
    min_distance: maksymalna różnica indeksów w jednej grupie
    """
    grouped = []
    current_group = [peaks[0]]

    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i-1] <= min_distance:
            current_group.append(peaks[i])
        else:
            grouped.append(current_group)
            current_group = [peaks[i]]
    grouped.append(current_group)

    # Wybierz najlepszy (największa wartość korelacji) z każdej grupy
    selected = []
    for group in grouped:
        best_index = group[np.argmax(corr[group])]
        selected.append(best_index)

    return np.array(selected)