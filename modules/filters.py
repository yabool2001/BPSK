import numpy as np
import time as t
import tomllib

from modules import modulation , plot
from numba import njit  # Dodane dla przyspieszenia obliczeń
from numpy.typing import NDArray
from pathlib import Path
from scipy.signal import lfilter , upfirdn , find_peaks

with open ( "settings.toml" , "rb" ) as settings_toml_file :
    toml_settings = tomllib.load ( settings_toml_file )

BETA = float ( toml_settings[ "rrc_filter" ][ "BETA" ] )
SPAN = int ( toml_settings[ "rrc_filter" ][ "SPAN" ] )
SPS  = int ( toml_settings[ "bpsk" ][ "SPS" ] )

def apply_tx_rrc_filter_v0_0_0 ( symbols: NDArray[ np.complex128 ] , upsample: bool = True , ) -> NDArray[ np.complex128 ] :
    """ Stosuje filtr Root Raised Cosine (RRC) z opcjonalnym upsamplingiem. Zawsze zwraca sygnał zespolony (complex128).
    Parametry:
        symbols: Sygnał wejściowy (real lub complex).
        beta: Współczynnik roll-off (0.0-1.0).
        sps: Próbek na symbol (samples per symbol).
        span: Długość filtra w symbolach.
        upsample: Czy wykonać upsampling (True) czy tylko filtrować (False).
    Zwraca:
        Przefiltrowany sygnał zespolony (complex128). """
    
    rrc_taps = rrc_filter_v0_0_0 ( beta = BETA , sps  = modulation.SPS , span = SPAN )
    
    if upsample:
        filtered = upfirdn ( rrc_taps , symbols , modulation.SPS )  # Auto-upsampling + filtracja
    else:
        filtered = lfilter ( rrc_taps , 1.0 , symbols )     # Tylko filtracja
    
    return ( filtered + 0j ) .astype ( np.complex128 )

def apply_rrc_rx_filter_v0_0_0 ( rx_samples: NDArray[ np.int16 ] ) -> NDArray[ np.int16 ] :
    """ Filtruje odebrane próbki z SDR filtrem RRC.
    Parametry: rx_samples: Odebrane próbki raw int16 (interleaved).
    Funkcja ma być wydajna w celu jej docelowego zaimplementowania na fizycznym modemie w FPGA lub DSP.
    Nie powinna operować na float tylko na int16 bez utraty precyzji i jakości filtrowania, czyli działać state of the art modem.
    Zwraca: Przefiltrowane próbki jako interleaved int16 (I, Q, I, Q...). """

    # Generuj współczynniki filtra RRC (float)
    rrc_taps_float = rrc_filter_v0_0_0 ( beta = BETA , sps  = SPS , span = SPAN )
    
    # Konwersja współczynników na stałoprzecinkowe (Q15 dla int16)
    # W praktyce DSP często mnoży się przez 2^15 (32768) dla Q15.
    scaling_factor = 32768.0 
    rrc_taps_fixed = np.round ( rrc_taps_float * scaling_factor ).astype ( np.int64 ) 
    
    # Rozdzielenie I/Q
    rx_i = rx_samples[ 0 : : 2 ].astype ( np.int64 )
    rx_q = rx_samples[ 1 : : 2 ].astype ( np.int64 )

    # Splot stałoprzecinkowy (symulowany bit-perfect na CPU)
    # Używamy np.convolve na int64, aby uniknąć przepełnienia akumulatora.
    filtered_real_accum = np.convolve ( rx_i , rrc_taps_fixed , mode = 'same' )
    filtered_imag_accum = np.convolve ( rx_q , rrc_taps_fixed , mode = 'same' )
    
    # Skalowanie w dół (bit shift) po filtracji
    filtered_real = ( ( filtered_real_accum + 16384 ) // 32768 ).astype ( np.int16 )
    filtered_imag = ( ( filtered_imag_accum + 16384 ) // 32768 ).astype ( np.int16 )
    
    # Ponowne splecenie (interleaving) do formatu int16
    result = np.empty ( filtered_real.size + filtered_imag.size , dtype = np.int16 )
    result[ 0 : : 2 ] = filtered_real
    result[ 1 : : 2 ] = filtered_imag
    
    return result

@njit ( cache = True , fastmath = True )  # Kompilacja Just-In-Time z optymalizacjami
def rrc_filter_v0_0_0 ( beta , sps , span ) :
    """ DeepSeek -V3 R1 (Zoptymalizowana)
        Filtruje dane wejściowe za pomocą filtra RRC. """
    
    N = span * sps
    t = np.arange ( -N / 2 , N / 2 + 1 , dtype = np.float64 ) / sps

    # Obsługa beta = 0 (filtr sinc)
    if beta == 0 :
        h = np.sinc ( t )
        h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja
        return h
    
    h = np.zeros_like ( t )

    # Stałe pre-kalkulowane dla wydajności
    beta_pi = np.pi * beta
    inv_4beta = 1 / ( 4 * beta )
    sqrt2 = np.sqrt ( 2 )
    special_val = ( beta / sqrt2 ) * \
                ( ( 1 + 2 / np.pi ) * np.sin ( np.pi / ( 4 * beta ) ) + 
                ( 1 - 2 / np.pi ) * np.cos ( np.pi / ( 4 * beta ) ) )

    # Obliczenia z maskami (zoptymalizowane)
    for i in range ( len ( t ) ) :
        ti = t[i]
        if ti == 0.0 :
            h[i] = 1.0 - beta + ( 4 * beta / np.pi )
        elif np.abs ( np.abs ( ti ) - inv_4beta ) < 1e-10 :  # Tolerancja numeryczna
            h[i] = special_val
        else :
            numerator = np.sin ( np.pi * ti * ( 1 - beta ) ) + \
                      4 * beta * ti * np.cos ( np.pi * ti * ( 1 + beta ) )
            denominator = np.pi * ti * ( 1 - ( 4 * beta * ti ) ** 2 )
            h[i] = numerator / denominator

    # Bezpieczna obsługa NaN/Inf i normalizacja
    h[ np.isnan ( h ) | np.isinf ( h ) ] = 0
    h = h / np.sqrt ( np.sum ( h ** 2 ) )  # Normalizacja

    return h
